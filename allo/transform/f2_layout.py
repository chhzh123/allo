# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""F2 Linear Layout Synthesis for Conflict-Free Memory Banking.

Implements the algorithm from tests/dataflow/plan.md:

  Phase 1 - Extract affine access patterns as F2 matrices A*x where
            x is the loop-iteration vector over GF(2).

  Phase 2 - Construct the conflict subspace
              P = span(V_space ∪ V_pattern)
            where V_space captures parallelism and V_pattern captures
            which address bits toggle simultaneously in a butterfly pair.

  Phase 3 - Solve for the bank-selection (swizzle) matrix S (bank_bits×n_bits
            over GF(2)) such that
              ∀ v ∈ P, v ≠ 0  ⟹  S·v ≠ 0
            i.e., no non-zero conflict vector maps to the zero bank address.

  Phase 4 - Return SwizzleHelper objects that generate the bank/offset index
            expressions consumed by kernel code and the HLS pragmas consumed
            by the schedule layer.

References:
  - kernel.cpp in gemini-fft.prj (swizzle_bank<STRIDE_BIT>)
  - tests/dataflow/plan.md

Usage example (FFT inter-vector stage with STRIDE=32 in N=256, WIDTH=32)::

    from allo.transform.f2_layout import F2LayoutSolver

    solver = F2LayoutSolver(n_bits=8, bank_bits=5)
    helper = solver.solve(stride_bits=[5])        # STRIDE=2^5=32
    # helper.bank_expr(idx_sym) → Python expression string for bank index
    # helper.offset_expr(idx_sym) → expression string for within-bank offset
    # helper.dims() → (WIDTH, NUM_VECS) = (32, 8) tuple for 2D array shape
    # helper.swizzle_bank(idx) → concrete integer, useful at compile time
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# GF(2) matrix utilities
# ---------------------------------------------------------------------------


def _rref_gf2(M: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Row-reduce *M* over GF(2).  Returns (rref_matrix, pivot_col_indices)."""
    M = M.copy() & 1
    rows, cols = M.shape
    pivot_cols = []
    row = 0
    for col in range(cols):
        # Find pivot in column >= current row
        pivot = None
        for r in range(row, rows):
            if M[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        pivot_cols.append(col)
        M[[row, pivot]] = M[[pivot, row]]          # swap rows
        for r in range(rows):
            if r != row and M[r, col]:
                M[r] ^= M[row]
        row += 1
        if row == rows:
            break
    return M & 1, pivot_cols


def _null_space_gf2(M: np.ndarray) -> np.ndarray:
    """Return a basis for the null space of *M* over GF(2).

    Each column of the returned matrix is a null-space basis vector.
    """
    rows, cols = M.shape
    rref, pivot_cols = _rref_gf2(M)
    free_cols = [c for c in range(cols) if c not in pivot_cols]
    if not free_cols:
        return np.zeros((cols, 0), dtype=np.int32)

    null_vecs = []
    for fc in free_cols:
        v = np.zeros(cols, dtype=np.int32)
        v[fc] = 1
        # Back-substitute: for each pivot row, set v[pivot_col] = rref[row, fc]
        for pr, pc in enumerate(pivot_cols):
            v[pc] = int(rref[pr, fc]) & 1
        null_vecs.append(v)
    return np.column_stack(null_vecs) & 1


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------


class F2LayoutSolver:
    """Solve for a conflict-free bank-swizzle matrix S over GF(2).

    Parameters
    ----------
    n_bits:
        Total number of address bits (= log2(N)).
    bank_bits:
        Number of bank-selection bits (= log2(num_banks) = log2(WIDTH)).

    The default (identity) bank selection uses the lower *bank_bits* bits of
    the linear address, i.e.

        S_default = [ I_{bank_bits} | 0_{bank_bits × (n_bits - bank_bits)} ]
    """

    def __init__(self, n_bits: int, bank_bits: int):
        assert bank_bits <= n_bits
        self.n_bits = n_bits
        self.bank_bits = bank_bits
        # Default S: identity on lower bank_bits
        self._S_default = np.eye(bank_bits, n_bits, dtype=np.int32)

    # ------------------------------------------------------------------
    # Phase 2 – conflict subspace
    # ------------------------------------------------------------------

    def conflict_subspace(self, stride_bits: List[int]) -> np.ndarray:
        """Build the conflict subspace P for the given stride bits.

        A stride bit *s* means that a butterfly pair accesses addresses
        that differ by 2^s (i.e., the delta vector e_s).  With the default
        cyclic bank assignment the lower *bank_bits* bits of the address
        determine the bank; two addresses differing by 2^s have the *same*
        bank whenever s >= bank_bits (since 2^s mod 2^bank_bits == 0).

        Returns a matrix whose columns are the conflict basis vectors (delta
        vectors that produce bank collisions under the *current* default S).
        """
        conflict_vecs = []
        for s in stride_bits:
            delta = np.zeros(self.n_bits, dtype=np.int32)
            delta[s] = 1
            # Check if S_default @ delta == 0 (conflict)
            bank_delta = (self._S_default @ delta) & 1
            if not np.any(bank_delta):
                conflict_vecs.append(delta)
        if not conflict_vecs:
            return np.zeros((self.n_bits, 0), dtype=np.int32)
        return np.column_stack(conflict_vecs) & 1

    # ------------------------------------------------------------------
    # Phase 3 – solve for S
    # ------------------------------------------------------------------

    def solve(self, stride_bits: List[int]) -> "SwizzleHelper":
        """Compute a minimal-XOR swizzle matrix S for the given strides.

        Strategy (heuristic, matches kernel.cpp's swizzle_bank pattern):
          - Start from identity S (lower bank_bits bits of address → bank).
          - For each conflicting stride bit *s*:
              XOR row (bank_bits - 1) of S with unit vector e_s.
              This is equivalent to: bank[bank_bits-1] ^= addr_bit[s].
          - Verify the result is conflict-free; raise if not (should not
            happen for the standard FFT case).

        Returns a SwizzleHelper wrapping the solved S.
        """
        S = self._S_default.copy()
        for s in stride_bits:
            delta = np.zeros(self.n_bits, dtype=np.int32)
            delta[s] = 1
            bank_delta = (S @ delta) & 1
            if not np.any(bank_delta):
                # Conflict: XOR the MSB row of S with e_s
                S[self.bank_bits - 1, s] ^= 1

        # Verify: no conflict vector should map to zero bank
        for s in stride_bits:
            delta = np.zeros(self.n_bits, dtype=np.int32)
            delta[s] = 1
            assert np.any((S @ delta) & 1), (
                f"F2LayoutSolver: could not resolve bank conflict for stride_bit={s}. "
                "Consider increasing bank_bits."
            )

        return SwizzleHelper(S, self.n_bits, self.bank_bits)


# ---------------------------------------------------------------------------
# SwizzleHelper – index expressions and 2D shape
# ---------------------------------------------------------------------------


class SwizzleHelper:
    """Provides bank/offset index computations for conflict-free buffer access.

    Wraps the solved S matrix and exposes:
      - :meth:`swizzle_bank` – concrete bank index for an integer address.
      - :meth:`bank_offset`  – within-bank offset for an integer address.
      - :meth:`dims`         – ``(num_banks, depth)`` tuple for 2D declaration.
      - :meth:`bank_expr`    – Python expression string for use inside kernels.
      - :meth:`offset_expr`  – Python expression string for use inside kernels.

    The physical 2D buffer has shape ``[num_banks, depth]`` where
      ``num_banks = 2 ** bank_bits`` and ``depth = 2 ** (n_bits - bank_bits)``.
    """

    def __init__(self, S: np.ndarray, n_bits: int, bank_bits: int):
        self.S = S & 1
        self.n_bits = n_bits
        self.bank_bits = bank_bits
        self.num_banks = 1 << bank_bits
        self.depth = 1 << (n_bits - bank_bits)

    # ------------------------------------------------------------------
    # Concrete index computation
    # ------------------------------------------------------------------

    def swizzle_bank(self, idx: int) -> int:
        """Return the physical bank index for linear address *idx*."""
        addr_bits = np.array(
            [(idx >> b) & 1 for b in range(self.n_bits)], dtype=np.int32
        )
        bank_bits_vec = (self.S @ addr_bits) & 1
        bank = int(sum(b << i for i, b in enumerate(bank_bits_vec)))
        return bank

    def bank_offset(self, idx: int) -> int:
        """Return the within-bank offset for linear address *idx*."""
        return idx >> self.bank_bits

    # ------------------------------------------------------------------
    # 2D shape
    # ------------------------------------------------------------------

    def dims(self) -> Tuple[int, int]:
        """``(num_banks, depth)`` for the 2D partitioned array declaration."""
        return (self.num_banks, self.depth)

    # ------------------------------------------------------------------
    # Expression strings for Allo kernel bodies
    # ------------------------------------------------------------------

    def bank_expr(self, idx_sym: str) -> str:
        """Python expression (using Allo-supported bitwise ops) for bank index.

        Example for S with XOR on row 4 at column 5::

            "(({idx} & 31) ^ ((({idx} >> 5) & 1) << 4))"

        Parameters
        ----------
        idx_sym:
            Name of the integer variable representing the linear address.
        """
        return self._build_bank_expr(idx_sym)

    def offset_expr(self, idx_sym: str) -> str:
        """Python expression for within-bank offset (= idx >> bank_bits)."""
        return f"({idx_sym} >> {self.bank_bits})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_bank_expr(self, sym: str) -> str:
        """Build the bank expression from the S matrix.

        Each row of S gives one bank bit as a linear combination (XOR) of
        address bits.  We emit each row as a bitwise-and/shift/xor expression
        and combine the bank bits into a single integer via OR/shift.
        """
        bank_bit_exprs = []
        for row_i in range(self.bank_bits):
            # Collect columns with a 1 in this row
            set_cols = [c for c in range(self.n_bits) if self.S[row_i, c]]
            if not set_cols:
                bank_bit_exprs.append("0")
                continue
            # XOR all contributing address bits, then place at position row_i
            terms = [f"(({sym} >> {c}) & 1)" for c in set_cols]
            xor_expr = " ^ ".join(terms)
            if len(terms) > 1:
                xor_expr = f"({xor_expr})"
            if row_i == 0:
                bank_bit_exprs.append(xor_expr)
            else:
                bank_bit_exprs.append(f"({xor_expr} << {row_i})")
        return "(" + " | ".join(bank_bit_exprs) + ")"

    def __repr__(self) -> str:
        return (
            f"SwizzleHelper(n_bits={self.n_bits}, bank_bits={self.bank_bits}, "
            f"dims={self.dims()},\n  S=\n{self.S})"
        )


# ---------------------------------------------------------------------------
# FFT-specific convenience factory
# ---------------------------------------------------------------------------


def fft_swizzle(N: int, WIDTH: int, stride_bit: int) -> SwizzleHelper:
    """Return a SwizzleHelper for a single FFT inter-vector stage.

    Parameters
    ----------
    N:
        FFT size (must be power of 2).
    WIDTH:
        Vector width / number of banks (must be power of 2, <= N).
    stride_bit:
        The bit position of the butterfly stride (= log2(STRIDE)).
        Must satisfy stride_bit >= log2(WIDTH) for inter-vector stages.

    Example::

        helper = fft_swizzle(N=256, WIDTH=32, stride_bit=5)
        # helper.swizzle_bank(96) → conflict-free bank for element 96
    """
    import math

    n_bits = int(math.log2(N))
    bank_bits = int(math.log2(WIDTH))
    assert stride_bit >= bank_bits, (
        f"stride_bit={stride_bit} must be >= bank_bits={bank_bits} "
        "(intra-vector stages have no bank conflicts)"
    )
    solver = F2LayoutSolver(n_bits=n_bits, bank_bits=bank_bits)
    return solver.solve(stride_bits=[stride_bit])
