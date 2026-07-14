# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module,too-many-function-args,unused-variable,unused-import,differing-param-doc,wrong-import-order,consider-using-alias

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
        M[[row, pivot]] = M[[pivot, row]]  # swap rows
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

    def _unit(self, s: int) -> np.ndarray:
        """The address delta e_s (a butterfly at stride bit *s* toggles address bit *s*)."""
        delta = np.zeros(self.n_bits, dtype=np.int32)
        delta[s] = 1
        return delta

    def solve(self, stride_bits: List[int]) -> "SwizzleHelper":
        """Compute a conflict-free swizzle matrix S for the given strides.

        A set of butterfly strides accessed together makes a whole conflict *subspace* live: the
        addresses that collide are every non-zero vector in ``span({e_s})``, not just the individual
        ``e_s``.  So each stride the default (low-bit) bank assignment does not already separate must
        toggle a **distinct** bank row -- if two shared one row, the combined delta ``e_{s_i} ^
        e_{s_j}`` would cancel to bank 0 (e.g. strides ``[2, 3]`` mapping ``{0,4,8,12}`` to the same
        bank).  This needs one bank bit per conflicting stride; **reject** when there are more
        conflicting strides than ``bank_bits`` (too few banks to separate that many simultaneous
        accesses), and verify the result separates the entire span before returning.

        For a single stride (the common FFT-stage case) this is exactly the original
        ``bank[bank_bits-1] ^= addr_bit[s]`` swizzle.

        Raises ``ValueError`` when the strides cannot be made conflict-free with ``bank_bits`` banks.
        """
        S = self._S_default.copy()
        strides = sorted(set(stride_bits))
        # A stride s < bank_bits is already its own bank bit (row s), so it "claims" that bank row;
        # a stride s >= bank_bits maps to bank 0 by default and needs a swizzle. Each such
        # conflicting stride must toggle a DISTINCT bank row that no other stride in the access set
        # already occupies -- otherwise two strides drive the same bank bit and their combined delta
        # cancels to bank 0 (e.g. stride 3 swizzled into the row bank bit 2 already uses).
        claimed = {s for s in strides if s < self.bank_bits}
        free_rows = [r for r in range(self.bank_bits - 1, -1, -1) if r not in claimed]
        conflicting = [s for s in strides if s >= self.bank_bits]
        if len(conflicting) > len(free_rows):
            raise ValueError(
                f"F2LayoutSolver: {len(strides)} simultaneous strides {strides} cannot be made "
                f"conflict-free with bank_bits={self.bank_bits} ({1 << self.bank_bits} banks); "
                f"need at least {1 << len(strides)} banks."
            )
        for row, s in zip(free_rows, conflicting):
            # top-down free rows: a single stride still lands in row bank_bits-1, the swizzle the
            # FFT kernel expects.
            S[row, s] ^= 1

        self._assert_conflict_free(S, strides)
        return SwizzleHelper(S, self.n_bits, self.bank_bits)

    def _assert_conflict_free(self, S: np.ndarray, stride_bits: List[int]) -> None:
        """Raise unless S separates every non-zero vector in the span of the stride deltas.

        S is conflict-free over ``span({e_s})`` iff the bank-delta images ``{S @ e_s}`` are linearly
        independent over GF(2) -- then no non-empty XOR of the strides can map to bank 0.  Checking
        rank is exact and O(#strides^2), unlike enumerating the 2^k span.
        """
        strides = sorted(set(stride_bits))
        if not strides:
            return
        images = np.array([(S @ self._unit(s)) & 1 for s in strides], dtype=np.int32)
        _, pivots = _rref_gf2(images)
        if len(pivots) < len(strides):
            raise ValueError(
                f"F2LayoutSolver: banking is not conflict-free for strides {strides} with "
                f"bank_bits={self.bank_bits}: some simultaneous access delta maps to bank 0. "
                f"Increase bank_bits."
            )

    def solve_subspace(self, P: np.ndarray) -> "SwizzleHelper":
        """Solve for a conflict-free swizzle from a GF(2) conflict *subspace basis* ``P`` directly.

        ``P`` is an ``n_bits x k`` matrix whose columns are the access deltas that must not collide
        (the differences of simultaneously-accessed addresses). Returns a ``SwizzleHelper`` whose
        bank matrix separates every non-zero vector in ``span(P)`` (``rank(S @ P) == rank(P)``), or
        raises ``ValueError`` when the bank count is too small or the current ``[I|T]`` swizzle (which
        keeps ``offset = idx >> bank_bits`` valid) cannot realize the layout. ``solve(stride_bits)``
        is the unit-stride convenience wrapper over this.
        """
        P = np.asarray(P, dtype=np.int32) & 1
        if P.ndim == 1:
            P = P.reshape(-1, 1)
        if P.shape[0] != self.n_bits:
            raise ValueError(
                f"F2LayoutSolver.solve_subspace: conflict basis has {P.shape[0]} address bits, "
                f"expected n_bits={self.n_bits}"
            )
        strides = []
        for col in P.T:
            nz = np.nonzero(col)[0]
            if len(nz) == 0:
                continue
            if len(nz) != 1:
                # The realized swizzle toggles single address bits into bank rows; a multi-bit
                # conflict delta would need a general [I|T] the emitter cannot yet realize. Reject
                # (fail closed) rather than emit a layout the HLS storage cannot honor.
                raise ValueError(
                    "F2LayoutSolver.solve_subspace realizes only unit-stride conflict bases "
                    f"(one set address bit per delta); got the multi-bit delta {col.tolist()}."
                )
            strides.append(int(nz[0]))
        return self.solve(strides)


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
        if self.bank_bits == 0:
            # a single bank -> always bank index 0 (avoids an empty "()" expression)
            return "0"
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


# ---------------------------------------------------------------------------
# MLIR transform: rewrite 1D buffer to 2D with F2 bank-swizzle indexing
# ---------------------------------------------------------------------------


def apply_f2_layout(
    module,
    func_name,
    buf_name,
    func_args,
    n_bits,
    bank_bits,
    stride_bit=None,
    banking="cyclic",
):
    """Transform a 1D buffer to 2D with bank-conflict-free indexing.

    Rewrites memref<N x f32> -> memref<num_banks x depth x f32>
    and replaces all load/store accesses with computed bank/offset indices.

    Parameters
    ----------
    module : MLIR Module
    func_name : str
    buf_name : str
    func_args : dict
    n_bits : int
        Total address bits (= log2(array_size)).
    bank_bits : int
        Bank selection bits (= log2(num_banks)).
    stride_bit : int or None
        Butterfly stride bit for F2 swizzle. Required when banking="cyclic".
    banking : str
        Banking mode:
        - "cyclic": bank = (addr & (W-1)) ^ (((addr >> stride_bit) & 1) << (bank_bits-1)),
                    offset = addr >> bank_bits.  (F2 XOR swizzle for butterfly strides)
        - "block":  bank = addr >> offset_bits, offset = addr & (depth-1).
                    (Upper bits select bank; conflict-free when writes are spread
                    across all N addresses and reads are sequential.)
    """
    from .._mlir.ir import (
        InsertionPoint,
        StringAttr,
        IndexType,
        IntegerType,
        IntegerAttr,
        MemRefType,
        Location,
        AffineMapAttr,
    )
    from .._mlir.dialects import (
        memref as memref_d,
        affine as affine_d,
        arith as arith_d,
        func as func_d,
    )
    from ..ir.utils import MockBuffer, MockArg
    from ..ir.transform import find_buffer

    num_banks = 1 << bank_bits
    depth = 1 << (n_bits - bank_bits)

    with module.context, Location.unknown():
        # 1. Find the old AllocOp
        target = MockBuffer(func_name, buf_name)
        func_op, _, old_alloc = find_buffer(module, target, func_args)

        # old_alloc should be a memref_d.AllocOp (not a MockArg)
        if isinstance(old_alloc, MockArg):
            return
        old_result = old_alloc.result
        elem_type = MemRefType(old_result.type).element_type

        # 1b. Pre-scan uses BEFORE mutating any IR. This transform only remaps load/store/partition
        # accesses onto the 2-D banked buffer; any OTHER use (an init fill, a copy, a stream put/get,
        # ...) means banking would be silently wrong. Checking first -- rather than after rewriting the
        # supported accesses -- guarantees a failure never leaves a half-rewritten schedule for the
        # caller to build on (`wrapped_apply` does not roll the Schedule back on a caught error).
        supported_ops = {
            "memref.load",
            "memref.store",
            "affine.load",
            "affine.store",
            "allo.partition",
        }
        unsupported = sorted(
            {
                use.owner.name
                for use in old_result.uses
                if hasattr(use.owner, "name") and use.owner.name not in supported_ops
            }
        )
        if unsupported:
            raise ValueError(
                f"apply_f2_layout: buffer {func_name}:{buf_name} has accesses this transform "
                f"cannot rewrite ({unsupported}); refusing to bank it. Only load/store/partition "
                f"accesses of the buffer are supported."
            )

        # 2. Create new 2D memref type: memref<num_banks x depth x elem_type>
        new_memref_type = MemRefType.get([num_banks, depth], elem_type)

        # 3. Create new AllocOp right after the old one
        ip = InsertionPoint.after(old_alloc.operation)
        new_alloc = memref_d.AllocOp(new_memref_type, [], [], ip=ip)
        new_alloc.attributes["name"] = StringAttr.get(buf_name)
        # Carry the `unsigned` marker (if any) onto the banked buffer, so an F2-banked `UInt` buffer
        # still prints as an unsigned/ap_uint storage in HLS rather than silently becoming signed.
        if "unsigned" in old_alloc.attributes:
            new_alloc.attributes["unsigned"] = old_alloc.attributes["unsigned"]

        # 4. Collect all uses of old alloc result before modifying
        uses = list(old_result.uses)

        i32 = IntegerType.get_signless(32)
        index_type = IndexType.get()

        for use in uses:
            op = use.owner
            if not hasattr(op, "name"):
                continue
            op_name = op.name

            if op_name == "memref.load":
                load_op = op.opview
                idx = load_op.indices[0]
                ip = InsertionPoint(op)

                bank_idx, offset_idx = _compute_bank_indices(
                    idx,
                    num_banks,
                    bank_bits,
                    stride_bit,
                    banking,
                    i32,
                    index_type,
                    ip,
                )

                new_load = memref_d.LoadOp(
                    new_alloc.result, [bank_idx, offset_idx], ip=ip
                )
                if "unsigned" in op.attributes:
                    new_load.operation.attributes["unsigned"] = op.attributes[
                        "unsigned"
                    ]
                load_op.result.replace_all_uses_with(new_load.result)
                op.erase()

            elif op_name == "memref.store":
                if use.operand_number < 1:
                    continue
                store_op = op.opview
                idx = store_op.indices[0]
                value = store_op.value
                ip = InsertionPoint(op)

                bank_idx, offset_idx = _compute_bank_indices(
                    idx,
                    num_banks,
                    bank_bits,
                    stride_bit,
                    banking,
                    i32,
                    index_type,
                    ip,
                )

                memref_d.StoreOp(value, new_alloc.result, [bank_idx, offset_idx], ip=ip)
                op.erase()

            elif op_name == "affine.load":
                load_op = op.opview
                ip = InsertionPoint(op)

                # Compute linear index using affine.apply with the same map
                amap = AffineMapAttr(op.attributes["map"]).value
                map_operands = list(load_op.indices)
                linear_idx = affine_d.AffineApplyOp(amap, map_operands, ip=ip)

                bank_idx, offset_idx = _compute_bank_indices(
                    linear_idx.result,
                    num_banks,
                    bank_bits,
                    stride_bit,
                    banking,
                    i32,
                    index_type,
                    ip,
                )

                new_load = memref_d.LoadOp(
                    new_alloc.result, [bank_idx, offset_idx], ip=ip
                )
                if "unsigned" in op.attributes:
                    new_load.operation.attributes["unsigned"] = op.attributes[
                        "unsigned"
                    ]
                load_op.result.replace_all_uses_with(new_load.result)
                op.erase()

            elif op_name == "affine.store":
                if use.operand_number < 1:
                    continue
                store_op = op.opview
                value = store_op.value
                ip = InsertionPoint(op)

                amap = AffineMapAttr(op.attributes["map"]).value
                map_operands = list(store_op.indices)
                linear_idx = affine_d.AffineApplyOp(amap, map_operands, ip=ip)

                bank_idx, offset_idx = _compute_bank_indices(
                    linear_idx.result,
                    num_banks,
                    bank_bits,
                    stride_bit,
                    banking,
                    i32,
                    index_type,
                    ip,
                )

                memref_d.StoreOp(value, new_alloc.result, [bank_idx, offset_idx], ip=ip)
                op.erase()

            elif op_name == "allo.partition":
                op.erase()

        # 5. Remove old alloc. Every load/store/partition was rewritten onto the 2D buffer above, and
        # the pre-scan (step 1b) already rejected any unsupported use before we mutated anything, so
        # the old buffer has no remaining uses here. This is a defensive backstop: if some use somehow
        # survives, raise rather than abort MLIR on erasing a live result.
        remaining = sorted({use.owner.name for use in old_result.uses})
        if remaining:
            raise ValueError(
                f"apply_f2_layout: buffer {func_name}:{buf_name} has accesses this transform "
                f"cannot rewrite ({remaining}); refusing to bank it. Only load/store/partition "
                f"accesses of the buffer are supported."
            )
        old_alloc.operation.erase()


def _compute_bank_indices(
    idx, num_banks, bank_bits, stride_bit, banking, i32, index_type, ip
):
    """Emit arith ops for bank/offset index computation.

    banking="cyclic" (F2 XOR swizzle):
        bank = (idx & (num_banks-1)) ^ (((idx >> stride_bit) & 1) << (bank_bits-1))
        offset = idx >> bank_bits

    banking="block":
        offset_bits = n_bits - bank_bits  (passed via stride_bit)
        bank = idx >> offset_bits
        offset = idx & (2^offset_bits - 1)
    """
    from .._mlir.dialects import arith as arith_d
    from .._mlir.ir import IntegerAttr

    # Avoid redundant index → i32 cast if idx is already from i32 → index cast
    defining_op = idx.owner
    if (
        defining_op is not None
        and hasattr(defining_op, "name")
        and defining_op.name == "arith.index_cast"
    ):
        src = defining_op.operands[0]
        if str(src.type) == "i32":
            idx_i32_val = src
        else:
            idx_i32_val = arith_d.IndexCastOp(i32, idx, ip=ip).result
    else:
        idx_i32_val = arith_d.IndexCastOp(i32, idx, ip=ip).result

    if banking == "block":
        offset_bits = stride_bit  # caller passes n_bits - bank_bits here
        depth = 1 << offset_bits

        shr_const = arith_d.ConstantOp(i32, IntegerAttr.get(i32, offset_bits), ip=ip)
        bank_i32 = arith_d.ShRUIOp(idx_i32_val, shr_const.result, ip=ip)

        depth_mask = arith_d.ConstantOp(i32, IntegerAttr.get(i32, depth - 1), ip=ip)
        offset_i32 = arith_d.AndIOp(idx_i32_val, depth_mask.result, ip=ip)
    else:
        # Compute offset first, then derive xor_bit from offset (not from
        # full idx).  This helps HLS because:
        #   idx = (offset << bank_bits) | raw_bank
        #   raw_bank = idx & (W-1)  → trivial when idx is composed via shift|or
        #   xor_bit  = (offset >> (stride_bit - bank_bits)) & 1
        #   bank     = raw_bank ^ (xor_bit << (bank_bits-1))
        # The lower bank_bits-1 bits of bank are always raw_bank[bank_bits-2:0],
        # making bank distinctness on the unrolled variable k trivially provable.

        # offset = idx >> bank_bits
        bank_bits_const = arith_d.ConstantOp(
            i32, IntegerAttr.get(i32, bank_bits), ip=ip
        )
        offset_i32 = arith_d.ShRUIOp(idx_i32_val, bank_bits_const.result, ip=ip)

        # raw_bank = idx & (num_banks - 1)
        mask_val = arith_d.ConstantOp(i32, IntegerAttr.get(i32, num_banks - 1), ip=ip)
        bank_low = arith_d.AndIOp(idx_i32_val, mask_val.result, ip=ip)

        if stride_bit is not None and stride_bit >= bank_bits:
            # XOR swizzle: xor_bit = (offset >> (stride_bit - bank_bits)) & 1
            rel_shift = stride_bit - bank_bits
            one_val = arith_d.ConstantOp(i32, IntegerAttr.get(i32, 1), ip=ip)
            if rel_shift > 0:
                rel_const = arith_d.ConstantOp(
                    i32, IntegerAttr.get(i32, rel_shift), ip=ip
                )
                shifted_off = arith_d.ShRUIOp(
                    offset_i32.result, rel_const.result, ip=ip
                )
                bit = arith_d.AndIOp(shifted_off.result, one_val.result, ip=ip)
            else:
                bit = arith_d.AndIOp(offset_i32.result, one_val.result, ip=ip)

            # bank = raw_bank ^ (xor_bit << (bank_bits - 1))
            shift_amt = arith_d.ConstantOp(
                i32, IntegerAttr.get(i32, bank_bits - 1), ip=ip
            )
            shifted_bit = arith_d.ShLIOp(bit.result, shift_amt.result, ip=ip)
            bank_i32 = arith_d.XOrIOp(bank_low.result, shifted_bit.result, ip=ip)
        else:
            # No XOR swizzle: plain cyclic partition
            # bank = idx & (num_banks - 1)
            bank_i32 = bank_low

    bank_idx = arith_d.IndexCastOp(index_type, bank_i32.result, ip=ip)
    offset_idx = arith_d.IndexCastOp(index_type, offset_i32.result, ip=ip)

    return bank_idx.result, offset_idx.result
