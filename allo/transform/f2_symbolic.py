# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""F2 Symbolic Execution Engine for automatic conflict subspace construction.

Implements symbolic tracking of bit-level dataflow over F2 (the binary field).
Each value is represented as an affine function of input variable bits:

    bits[i] = (sum_j matrix[i][j] * input_var_bits[j]) XOR constant[i]

where all arithmetic is mod 2 (XOR for addition, AND for multiplication).

Operations that are linear over F2 (XOR, shift by constant, AND with constant,
non-overlapping OR/add) are tracked exactly. Non-linear operations (AND of two
symbolic values, add with carries) produce conservative "opaque" symbols.

The conflict subspace P is built from two sources:
  1. Intra-iteration: constant/matrix deltas between simultaneous accesses
  2. Inter-iteration: columns at unrolled variable bit positions

References:
  - tests/dataflow/auto_f2_plan.md
  - allo/transform/f2_layout.py
"""

import numpy as np


class F2Symbol:
    """Symbolic bit vector over F2.

    Represents a value as an affine function:
        bits[i] = (sum_j matrix[i][j] * input_var_bits[j]) XOR constant[i]

    Attributes:
        matrix: numpy uint8 array of shape (n_bits, n_input_vars).
            Row i, column j indicates whether output bit i depends on input bit j.
        constant: numpy uint8 array of shape (n_bits,).
            The constant (affine) offset for each output bit.
        n_bits: number of output bits in this symbol.
    """

    def __init__(self, matrix, constant, n_bits):
        assert matrix.shape[0] == n_bits
        assert constant.shape[0] == n_bits
        self.matrix = matrix.astype(np.uint8)
        self.constant = constant.astype(np.uint8)
        self.n_bits = n_bits

    @property
    def n_input_vars(self):
        """Number of input variable bits tracked."""
        return self.matrix.shape[1]

    @classmethod
    def variable(cls, start_bit, var_n_bits, total_input_bits, n_output_bits):
        """Create symbol for a variable occupying bits [start_bit, start_bit+var_n_bits).

        The variable's bit k maps to output bit k (for k < var_n_bits), with
        identity columns at positions start_bit..start_bit+var_n_bits-1 in the
        input space.

        Args:
            start_bit: starting column index in the input bit vector.
            var_n_bits: number of bits this variable occupies.
            total_input_bits: total width of the input bit vector.
            n_output_bits: number of output bits in the result symbol.

        Returns:
            F2Symbol with identity mapping for the variable's bits.
        """
        matrix = np.zeros((n_output_bits, total_input_bits), dtype=np.uint8)
        for k in range(min(var_n_bits, n_output_bits)):
            if start_bit + k < total_input_bits:
                matrix[k, start_bit + k] = 1
        constant = np.zeros(n_output_bits, dtype=np.uint8)
        return cls(matrix, constant, n_output_bits)

    @classmethod
    def constant_val(cls, value, n_output_bits, total_input_bits):
        """Create symbol for a compile-time constant.

        Args:
            value: integer constant value.
            n_output_bits: number of output bits.
            total_input_bits: total width of the input bit vector.

        Returns:
            F2Symbol with zero matrix and bit-decomposed constant.
        """
        matrix = np.zeros((n_output_bits, total_input_bits), dtype=np.uint8)
        constant = np.zeros(n_output_bits, dtype=np.uint8)
        for i in range(n_output_bits):
            constant[i] = (value >> i) & 1
        return cls(matrix, constant, n_output_bits)

    @classmethod
    def opaque(cls, n_output_bits, total_input_bits):
        """Create an opaque symbol for conservative analysis.

        Used when a non-linear operation is encountered. All output bits are
        treated as independent — the matrix is set to an identity-like pattern
        so that every output bit depends on a distinct input bit (up to the
        minimum of n_output_bits and total_input_bits).

        This is conservative: any column extraction will find nonzero entries,
        forcing the conflict subspace to grow maximally.

        Returns:
            F2Symbol with identity-like matrix (conservative).
        """
        matrix = np.zeros((n_output_bits, total_input_bits), dtype=np.uint8)
        for i in range(min(n_output_bits, total_input_bits)):
            matrix[i, i] = 1
        constant = np.zeros(n_output_bits, dtype=np.uint8)
        return cls(matrix, constant, n_output_bits)

    def xor(self, other):
        """Compute self XOR other (F2 addition).

        Both symbols must have the same n_bits and n_input_vars.

        Returns:
            F2Symbol with element-wise XOR of matrices and constants.
        """
        assert self.n_bits == other.n_bits
        assert self.n_input_vars == other.n_input_vars
        return F2Symbol(
            self.matrix ^ other.matrix,
            self.constant ^ other.constant,
            self.n_bits,
        )

    def shift_left(self, k):
        """Compute self << k (shift left by constant k).

        In bit indexing where bit 0 is LSB: shifting left by k moves bit i
        to bit i+k. Rows shift upward: new_matrix[i+k] = old_matrix[i].

        Returns:
            F2Symbol with shifted rows, zero-filled at the bottom k rows.
        """
        if k == 0:
            return F2Symbol(self.matrix.copy(), self.constant.copy(), self.n_bits)
        if k >= self.n_bits:
            return F2Symbol(
                np.zeros_like(self.matrix),
                np.zeros_like(self.constant),
                self.n_bits,
            )
        new_matrix = np.zeros_like(self.matrix)
        new_constant = np.zeros_like(self.constant)
        new_matrix[k:] = self.matrix[: self.n_bits - k]
        new_constant[k:] = self.constant[: self.n_bits - k]
        return F2Symbol(new_matrix, new_constant, self.n_bits)

    def shift_right(self, k):
        """Compute self >> k (logical shift right by constant k).

        Moves bit i+k to bit i. Rows shift downward: new_matrix[i] = old_matrix[i+k].

        Returns:
            F2Symbol with shifted rows, zero-filled at the top k rows.
        """
        if k == 0:
            return F2Symbol(self.matrix.copy(), self.constant.copy(), self.n_bits)
        if k >= self.n_bits:
            return F2Symbol(
                np.zeros_like(self.matrix),
                np.zeros_like(self.constant),
                self.n_bits,
            )
        new_matrix = np.zeros_like(self.matrix)
        new_constant = np.zeros_like(self.constant)
        new_matrix[: self.n_bits - k] = self.matrix[k:]
        new_constant[: self.n_bits - k] = self.constant[k:]
        return F2Symbol(new_matrix, new_constant, self.n_bits)

    def and_constant(self, mask):
        """Compute self & mask where mask is an integer constant.

        Zeros out rows where the corresponding mask bit is 0.

        Returns:
            F2Symbol with masked rows.
        """
        new_matrix = self.matrix.copy()
        new_constant = self.constant.copy()
        for i in range(self.n_bits):
            if not ((mask >> i) & 1):
                new_matrix[i, :] = 0
                new_constant[i] = 0
        return F2Symbol(new_matrix, new_constant, self.n_bits)

    def is_nonoverlapping(self, other):
        """Check if self and other have non-overlapping active bits.

        Two symbols are non-overlapping if no output bit position has nonzero
        entries in both symbols (considering both matrix rows and constants).
        When non-overlapping, OR and addition are equivalent to XOR.

        Returns:
            True if the symbols are non-overlapping.
        """
        assert self.n_bits == other.n_bits
        for i in range(self.n_bits):
            self_active = self.matrix[i].any() or self.constant[i]
            other_active = other.matrix[i].any() or other.constant[i]
            if self_active and other_active:
                return False
        return True

    def or_nonoverlapping(self, other):
        """Compute self | other when bits are known to be non-overlapping.

        Equivalent to XOR when bits don't overlap (no carries possible).

        Returns:
            F2Symbol via XOR (safe when non-overlapping).
        """
        return self.xor(other)

    def truncate(self, n_bits):
        """Keep only the lower n_bits.

        Returns:
            F2Symbol with rows beyond n_bits discarded.
        """
        assert n_bits <= self.n_bits
        return F2Symbol(
            self.matrix[:n_bits].copy(),
            self.constant[:n_bits].copy(),
            n_bits,
        )

    def conflict_columns(self, var_bit_indices):
        """Extract sub-matrix columns for given input bit indices.

        These columns span the set of address-bit differences when the
        specified input variable ranges over all values.

        Args:
            var_bit_indices: list of column indices to extract.

        Returns:
            numpy array of shape (n_bits, len(var_bit_indices)).
        """
        return self.matrix[:, var_bit_indices].copy()

    def __repr__(self):
        return (
            f"F2Symbol(n_bits={self.n_bits}, n_input={self.n_input_vars}, "
            f"const={self.constant.tolist()})"
        )


def row_reduce_f2(vectors):
    """Gaussian elimination over F2 to find linearly independent columns.

    Performs column-wise reduction: treats each column of the input as a vector,
    and finds a maximal set of linearly independent columns (a basis for the
    column space).

    Args:
        vectors: numpy uint8 array of shape (n_bits, n_vectors).
            Each column is a vector over F2.

    Returns:
        numpy uint8 array of shape (n_bits, rank) containing linearly
        independent columns forming a basis of the column space.
    """
    if vectors.size == 0:
        return vectors.copy()

    n_rows, n_cols = vectors.shape
    # Work with a copy, transpose to do row reduction on rows = original columns
    # Actually, we do standard row echelon form on the matrix to find pivot columns.
    mat = vectors.copy() % 2

    pivot_cols = []
    pivot_row = 0
    for col in range(n_cols):
        # Find a row with a 1 in this column at or below pivot_row
        found = -1
        for row in range(pivot_row, n_rows):
            if mat[row, col]:
                found = row
                break
        if found == -1:
            continue
        # Swap rows
        if found != pivot_row:
            mat[[pivot_row, found]] = mat[[found, pivot_row]]
        # Eliminate this column in all other rows
        for row in range(n_rows):
            if row != pivot_row and mat[row, col]:
                mat[row] ^= mat[pivot_row]
        pivot_cols.append(col)
        pivot_row += 1

    # The pivot columns of the original matrix form a basis
    if not pivot_cols:
        return np.zeros((n_rows, 0), dtype=np.uint8)
    return vectors[:, pivot_cols].copy() % 2


def build_conflict_subspace(symbols, parallel_var_bits):
    """Construct the conflict subspace P from symbolic access expressions.

    The conflict subspace captures ALL address differences between accesses
    that happen in the SAME clock cycle. Two sources of parallelism:

    1. Intra-iteration: multiple load/store ops in the same loop body
       (always parallel, even in pipelined loops). The XOR of every pair
       of symbols gives constant and matrix deltas.
    2. Inter-iteration: unrolled loop iterations that execute simultaneously
       (from folding or explicit unrolling). The columns of each symbol's
       matrix at the parallel variable bit positions.

    Args:
        symbols: list of F2Symbol objects, one per memory access expression.
        parallel_var_bits: list of input bit indices for unrolled/folded
            variables. Empty list if the loop is pipelined (not unrolled).

    Returns:
        P: numpy uint8 array of shape (n_bits, rank) — basis vectors of
           the conflict subspace as columns.
    """
    if not symbols:
        return np.zeros((0, 0), dtype=np.uint8)

    n_bits = symbols[0].n_bits
    conflict_vectors = []

    # SOURCE 1: Intra-iteration conflicts (all pairs of accesses)
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            sym_a = symbols[i]
            sym_b = symbols[j]

            # Constant part: fixed address difference
            delta_const = sym_a.constant ^ sym_b.constant
            if delta_const.any():
                conflict_vectors.append(delta_const)

            # Matrix part: variable-dependent address differences
            delta_matrix = sym_a.matrix ^ sym_b.matrix
            for col_idx in range(delta_matrix.shape[1]):
                col = delta_matrix[:, col_idx]
                if col.any():
                    conflict_vectors.append(col)

    # SOURCE 2: Inter-iteration conflicts (only if loop is unrolled)
    if parallel_var_bits:
        for sym in symbols:
            for bit_idx in parallel_var_bits:
                col = sym.matrix[:, bit_idx]
                if col.any():
                    conflict_vectors.append(col.copy())

    # Row-reduce to get basis of P
    if conflict_vectors:
        all_vecs = np.column_stack(conflict_vectors)
        P = row_reduce_f2(all_vecs)
    else:
        P = np.zeros((n_bits, 0), dtype=np.uint8)
    return P


if __name__ == "__main__":
    print("=" * 60)
    print("F2 Symbolic Execution Engine — Unit Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    def check(name, condition):
        global passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}")
            failed += 1

    # ----------------------------------------------------------------
    # Test 1: Cyclic partition — A[i*2] and A[i*2+1], pipelined loop
    # ----------------------------------------------------------------
    print("\nTest 1: Cyclic partition (pipelined loop)")
    print("  A[i*2] and A[i*2+1] with i pipelined (not unrolled)")
    # i has 4 bits at positions 0..3, total_input_bits=4, n_output_bits=5
    # (5 addr bits for array of size 32)
    n_addr = 5
    n_input = 4

    # sym(i*2) = i << 1
    sym_i = F2Symbol.variable(0, 4, n_input, n_addr)
    sym_i2 = sym_i.shift_left(1)

    # sym(i*2+1) = (i << 1) | 1 = (i << 1) XOR 1
    sym_one = F2Symbol.constant_val(1, n_addr, n_input)
    sym_i2p1 = sym_i2.xor(sym_one)

    # Verify non-overlapping for (i<<1) and constant 1
    check("i<<1 and 1 are non-overlapping", sym_i2.is_nonoverlapping(sym_one))

    # Build conflict subspace — pipelined, no parallel vars
    P = build_conflict_subspace([sym_i2, sym_i2p1], parallel_var_bits=[])

    # Expect P = span(e_0), dim=1
    check("P has rank 1", P.shape[1] == 1)
    e0 = np.zeros(n_addr, dtype=np.uint8)
    e0[0] = 1
    check("P spans e_0", np.array_equal(P[:, 0], e0))

    # ----------------------------------------------------------------
    # Test 2: Same accesses but with unrolled loop
    # ----------------------------------------------------------------
    print("\nTest 2: Cyclic partition (unrolled loop)")
    print("  A[i*2] and A[i*2+1] with i unrolled (2-bit i)")
    n_addr2 = 4
    n_input2 = 2  # i has 2 bits

    sym_i_u = F2Symbol.variable(0, 2, n_input2, n_addr2)
    sym_i2_u = sym_i_u.shift_left(1)
    sym_one_u = F2Symbol.constant_val(1, n_addr2, n_input2)
    sym_i2p1_u = sym_i2_u.xor(sym_one_u)

    # Unrolled: parallel_var_bits = [0, 1] (both bits of i)
    P2 = build_conflict_subspace([sym_i2_u, sym_i2p1_u], parallel_var_bits=[0, 1])

    # Expect P = span(e_0, e_1, e_2), dim=3 (need 8 banks for 4 iters * 2 accesses)
    # Actually: Source 1 gives e_0 (delta_const). Source 2 gives columns for i bits
    # in sym(i*2): col 0 = e_1, col 1 = e_2. In sym(i*2+1): col 0 = e_1, col 1 = e_2.
    # So vectors: e_0, e_1, e_2 → rank 3
    check("P has rank 3 (unrolled)", P2.shape[1] == 3)

    # Check that e_0, e_1, e_2 are all in the span
    def in_span(vec, basis):
        """Check if vec is in the column span of basis over F2."""
        if basis.shape[1] == 0:
            return not vec.any()
        aug = np.column_stack([basis, vec])
        rank_basis = row_reduce_f2(basis).shape[1]
        rank_aug = row_reduce_f2(aug).shape[1]
        return rank_aug == rank_basis

    for bit in range(3):
        ev = np.zeros(n_addr2, dtype=np.uint8)
        ev[bit] = 1
        check(f"e_{bit} in span of P (unrolled)", in_span(ev, P2))

    # ----------------------------------------------------------------
    # Test 3: FFT butterfly — upper and lower = upper XOR stride
    # ----------------------------------------------------------------
    print("\nTest 3: FFT butterfly (upper XOR stride)")
    print("  upper = b (7 bits), lower = b XOR stride, stride = 32 = bit 5")
    n_addr3 = 8  # 256-element array
    n_input3 = 7  # b has 7 bits (b_inner=4, b_outer=3)

    # upper = b (identity mapping for simplicity)
    sym_upper = F2Symbol.variable(0, 7, n_input3, n_addr3)

    # lower = upper XOR 32 (stride = 2^5)
    sym_stride = F2Symbol.constant_val(32, n_addr3, n_input3)
    sym_lower = sym_upper.xor(sym_stride)

    # Pipelined (no unrolling): only intra-iteration conflict
    P3 = build_conflict_subspace([sym_upper, sym_lower], parallel_var_bits=[])

    check("P has rank 1 (stride)", P3.shape[1] == 1)
    e5 = np.zeros(n_addr3, dtype=np.uint8)
    e5[5] = 1
    check("P spans e_5 (stride bit)", np.array_equal(P3[:, 0], e5))

    # With b_inner (4 bits) unrolled
    P3u = build_conflict_subspace(
        [sym_upper, sym_lower], parallel_var_bits=[0, 1, 2, 3]
    )
    check("P has rank 5 (stride + 4 unrolled bits)", P3u.shape[1] == 5)
    for bit in [0, 1, 2, 3, 5]:
        ev = np.zeros(n_addr3, dtype=np.uint8)
        ev[bit] = 1
        check(f"e_{bit} in span of P (FFT unrolled)", in_span(ev, P3u))

    # ----------------------------------------------------------------
    # Test 4: Block access — B[i * BLOCK_SIZE + k] with k unrolled
    # ----------------------------------------------------------------
    print("\nTest 4: Block access pattern")
    print("  B[i * 4 + k] with BLOCK_SIZE=4, k unrolled (2 bits)")
    BLOCK_SIZE = 4
    n_addr4 = 6  # 64-element array
    # Input: k has 2 bits at positions 0,1; i has 4 bits at positions 2..5
    n_input4 = 6

    sym_i4 = F2Symbol.variable(2, 4, n_input4, n_addr4)
    sym_k4 = F2Symbol.variable(0, 2, n_input4, n_addr4)

    # i * 4 = i << 2
    sym_i_shifted = sym_i4.shift_left(2)

    # i*4 + k: non-overlapping (i<<2 uses bits 2+, k uses bits 0,1)
    check("i<<2 and k are non-overlapping", sym_i_shifted.is_nonoverlapping(sym_k4))
    sym_addr4 = sym_i_shifted.or_nonoverlapping(sym_k4)

    # k is unrolled: parallel_var_bits = [0, 1]
    P4 = build_conflict_subspace([sym_addr4], parallel_var_bits=[0, 1])

    check("P has rank 2 (k bits)", P4.shape[1] == 2)
    for bit in [0, 1]:
        ev = np.zeros(n_addr4, dtype=np.uint8)
        ev[bit] = 1
        check(f"e_{bit} in span of P (block access)", in_span(ev, P4))

    # ----------------------------------------------------------------
    # Test 5: F2Symbol basic operations
    # ----------------------------------------------------------------
    print("\nTest 5: Basic operations")

    # Constant creation
    c42 = F2Symbol.constant_val(42, 8, 4)
    check("constant(42) bit 1 = 1", c42.constant[1] == 1)
    check("constant(42) bit 3 = 1", c42.constant[3] == 1)
    check("constant(42) bit 5 = 1", c42.constant[5] == 1)
    check("constant(42) bit 0 = 0", c42.constant[0] == 0)
    check("constant(42) matrix is zero", not c42.matrix.any())

    # Shift right
    sym_v = F2Symbol.variable(0, 4, 4, 8)
    sym_sr = sym_v.shift_right(2)
    # bit 0 of result = bit 2 of input → matrix[0, 2] = 1
    check("shift_right(2): bit 0 depends on input 2", sym_sr.matrix[0, 2] == 1)
    check("shift_right(2): bit 1 depends on input 3", sym_sr.matrix[1, 3] == 1)
    check("shift_right(2): bits 2+ are zero", not sym_sr.matrix[2:].any())

    # AND with constant mask
    sym_mask = sym_v.and_constant(0b0101)  # keep bits 0 and 2
    check("and_constant: bit 0 preserved", sym_mask.matrix[0, 0] == 1)
    check("and_constant: bit 1 zeroed", not sym_mask.matrix[1].any())
    check("and_constant: bit 2 preserved", sym_mask.matrix[2, 2] == 1)
    check("and_constant: bit 3 zeroed", not sym_mask.matrix[3].any())

    # Truncate
    sym_trunc = sym_v.truncate(3)
    check("truncate: n_bits reduced", sym_trunc.n_bits == 3)
    check("truncate: matrix shape correct", sym_trunc.matrix.shape == (3, 4))

    # Opaque
    sym_opaque = F2Symbol.opaque(4, 6)
    check("opaque: identity-like", sym_opaque.matrix[0, 0] == 1)
    check("opaque: identity-like", sym_opaque.matrix[3, 3] == 1)
    check("opaque: off-diagonal zero", sym_opaque.matrix[0, 1] == 0)

    # ----------------------------------------------------------------
    # Test 6: row_reduce_f2
    # ----------------------------------------------------------------
    print("\nTest 6: row_reduce_f2")

    # Redundant vectors: e_0, e_1, e_0 XOR e_1
    vecs = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=np.uint8)
    basis = row_reduce_f2(vecs)
    check("rank of {e0, e1, e0^e1} = 2", basis.shape[1] == 2)

    # All zeros
    zvecs = np.zeros((3, 3), dtype=np.uint8)
    zbasis = row_reduce_f2(zvecs)
    check("rank of zero matrix = 0", zbasis.shape[1] == 0)

    # Full rank identity
    eye = np.eye(4, dtype=np.uint8)
    ebasis = row_reduce_f2(eye)
    check("rank of I_4 = 4", ebasis.shape[1] == 4)

    # ----------------------------------------------------------------
    # Test 7: FFT inter-stage index expression (from plan)
    # ----------------------------------------------------------------
    print("\nTest 7: FFT inter-stage composite index expression")
    print("  upper = (group << 6) | pos, group = b >> 5, pos = b & 31")
    print("  lower = upper XOR 32, with b_inner (4 bits) unrolled")
    n_addr7 = 8
    # b_inner: 4 bits at [0..3], b_outer: 3 bits at [4..6]
    n_input7 = 7

    sym_b = F2Symbol.variable(0, 7, n_input7, n_addr7)

    # group = b >> 5 (shift right by 5)
    sym_group = sym_b.shift_right(5)

    # pos = b & 31 (mask lower 5 bits)
    sym_pos = sym_b.and_constant(31)

    # upper = (group << 6) | pos
    sym_group_shifted = sym_group.shift_left(6)
    check(
        "group<<6 and pos are non-overlapping",
        sym_group_shifted.is_nonoverlapping(sym_pos),
    )
    sym_upper7 = sym_group_shifted.or_nonoverlapping(sym_pos)

    # lower = upper XOR 32
    sym_stride7 = F2Symbol.constant_val(32, n_addr7, n_input7)
    sym_lower7 = sym_upper7.xor(sym_stride7)

    # Verify upper address mapping:
    # addr[0..4] = b[0..4] (pos), addr[5] = 0, addr[6] = b[5], addr[7] = b[6]
    check("upper addr[0] = b[0]", sym_upper7.matrix[0, 0] == 1)
    check("upper addr[4] = b[4]", sym_upper7.matrix[4, 4] == 1)
    check("upper addr[5] = 0 (gap)", not sym_upper7.matrix[5].any())
    check("upper addr[6] = b[5]", sym_upper7.matrix[6, 5] == 1)
    check("upper addr[7] = b[6]", sym_upper7.matrix[7, 6] == 1)

    # lower differs at bit 5 (stride)
    check(
        "lower const[5] = 1 (stride)",
        sym_lower7.constant[5] == 1,
    )

    # Conflict subspace with b_inner (bits 0..3) unrolled
    P7 = build_conflict_subspace(
        [sym_upper7, sym_lower7], parallel_var_bits=[0, 1, 2, 3]
    )

    check("P has rank 5", P7.shape[1] == 5)
    for bit in [0, 1, 2, 3, 5]:
        ev = np.zeros(n_addr7, dtype=np.uint8)
        ev[bit] = 1
        check(f"e_{bit} in span of P (FFT inter-stage)", in_span(ev, P7))

    # e_4 should NOT be in P (b_outer[0] is not unrolled, same in both accesses)
    e4 = np.zeros(n_addr7, dtype=np.uint8)
    e4[4] = 1
    check("e_4 NOT in P (sequential var)", not in_span(e4, P7))

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"WARNING: {failed} tests failed!")
    print("=" * 60)
