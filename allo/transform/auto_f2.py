# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic F2 bank-conflict-free layout analysis and application.

Walks MLIR SSA chains to build F2Symbol representations of index expressions,
then uses conflict subspace analysis to determine optimal partitioning.

Usage:
    from allo.transform.auto_f2 import auto_apply_f2
    auto_apply_f2(module, func_name, func_args)

References:
    - allo/transform/f2_symbolic.py (F2Symbol engine)
    - allo/transform/f2_layout.py (F2LayoutSolver, apply_f2_layout)
    - tests/dataflow/auto_f2_plan.md (design document)
"""

import math
import numpy as np

from .f2_symbolic import F2Symbol, build_conflict_subspace, row_reduce_f2
from .f2_layout import apply_f2_layout


# ---------------------------------------------------------------------------
# SSA value → F2Symbol conversion
# ---------------------------------------------------------------------------


def _get_constant_int(value):
    """Extract an integer constant from an MLIR SSA value, or return None.

    Traces through:
    - arith.constant → direct integer value
    - arith.index_cast / arith.extui / arith.extsi / arith.trunci → recurse
    - affine.load / memref.load of scalar memref → find store, recurse
    """
    try:
        op = value.owner
        if op is None:
            return None
    except Exception:
        return None
    try:
        op_name = op.name
        if op_name == "arith.constant":
            from .._mlir.ir import IntegerAttr as MlirIntegerAttr
            attr = MlirIntegerAttr(op.attributes["value"])
            return attr.value
        if op_name in ("arith.index_cast", "arith.extui", "arith.extsi",
                       "arith.trunci"):
            return _get_constant_int(op.operands[0])
        if op_name == "affine.load":
            memref_val = op.operands[0]
            stored = _find_scalar_store_value(memref_val, op)
            if stored is not None:
                return _get_constant_int(stored)
        if op_name == "memref.load" and len(list(op.operands)) == 1:
            memref_val = op.operands[0]
            stored = _find_scalar_store_value(memref_val, op)
            if stored is not None:
                return _get_constant_int(stored)
    except Exception:
        pass
    return None


def _is_power_of_2(n):
    """Check if n is a positive power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def _find_scalar_store_value(memref_val, load_op):
    """Find the value stored to a scalar memref that feeds a given load.

    Allo lowers variable assignments like `il: uint32 = expr` into:
        %alloc = memref.alloc() : memref<i32>
        affine.store %expr, %alloc[] : memref<i32>
        ...
        %val = affine.load %alloc[] : memref<i32>

    This function finds the store and returns the stored value SSA.

    Args:
        memref_val: the scalar memref SSA value (from alloc).
        load_op: the load operation (for dominance ordering).

    Returns:
        The SSA value that was stored, or None if not found.
    """
    # Walk all uses of the memref value to find stores
    try:
        uses = list(memref_val.uses)
    except Exception:
        return None

    store_val = None
    for use in uses:
        user_op = use.owner
        user_name = user_op.name
        if user_name == "affine.store" and use.operand_number == 1:
            # operand 0 is value, operand 1 is memref for affine.store
            store_val = user_op.operands[0]
        elif user_name == "memref.store" and use.operand_number == 1:
            # operand 0 is value, operand 1 is memref for memref.store
            store_val = user_op.operands[0]

    return store_val


def symbolize_mlir_index(ssa_value, loop_var_map, n_addr_bits, n_input_bits,
                         memo=None):
    """Convert an MLIR SSA value to an F2Symbol by walking the SSA def-use chain.

    Args:
        ssa_value: MLIR SSA value (the array index expression).
        loop_var_map: dict mapping MLIR SSA values to (start_bit, n_bits) tuples
            describing their position in the F2 input bit vector.
        n_addr_bits: number of address bits (log2 of array size).
        n_input_bits: total number of input variable bits.
        memo: optional dict for memoization (avoids re-symbolizing shared SSA values).

    Returns:
        F2Symbol representing the index expression.
    """
    if memo is None:
        memo = {}

    # Use the SSA value directly as memo key (MLIR values are hashable).
    # Do NOT use id() — Python may reuse id() after GC of temporary wrappers,
    # causing stale memo hits.
    try:
        if ssa_value in memo:
            return memo[ssa_value]
    except Exception:
        pass

    result = _symbolize_impl(ssa_value, loop_var_map, n_addr_bits,
                             n_input_bits, memo)
    try:
        memo[ssa_value] = result
    except Exception:
        pass
    return result


def _symbolize_impl(ssa_value, loop_var_map, n_addr_bits, n_input_bits, memo):
    """Internal implementation of symbolize_mlir_index."""
    # Check if this is a known loop variable
    for var_ssa, (start_bit, n_bits) in loop_var_map.items():
        if ssa_value == var_ssa:
            return F2Symbol.variable(start_bit, n_bits, n_input_bits,
                                     n_addr_bits)

    # Try to get the defining operation
    try:
        op = ssa_value.owner
        if op is None:
            return F2Symbol.opaque(n_addr_bits, n_input_bits)
    except Exception:
        return F2Symbol.opaque(n_addr_bits, n_input_bits)

    op_name = op.name

    # arith.constant → constant value
    if op_name == "arith.constant":
        val = _get_constant_int(ssa_value)
        if val is not None:
            return F2Symbol.constant_val(val, n_addr_bits, n_input_bits)
        return F2Symbol.opaque(n_addr_bits, n_input_bits)

    # arith.xori → lhs XOR rhs
    if op_name == "arith.xori":
        lhs = symbolize_mlir_index(op.operands[0], loop_var_map, n_addr_bits,
                                   n_input_bits, memo)
        rhs = symbolize_mlir_index(op.operands[1], loop_var_map, n_addr_bits,
                                   n_input_bits, memo)
        return lhs.xor(rhs)

    # arith.shli → shift left by constant
    if op_name == "arith.shli":
        shift = _get_constant_int(op.operands[1])
        if shift is not None:
            val = symbolize_mlir_index(op.operands[0], loop_var_map,
                                       n_addr_bits, n_input_bits, memo)
            return val.shift_left(shift)
        return F2Symbol.opaque(n_addr_bits, n_input_bits)

    # arith.shrui / arith.shrsi → shift right by constant
    if op_name in ("arith.shrui", "arith.shrsi"):
        shift = _get_constant_int(op.operands[1])
        if shift is not None:
            val = symbolize_mlir_index(op.operands[0], loop_var_map,
                                       n_addr_bits, n_input_bits, memo)
            return val.shift_right(shift)
        return F2Symbol.opaque(n_addr_bits, n_input_bits)

    # arith.andi → AND with constant mask
    if op_name == "arith.andi":
        c0 = _get_constant_int(op.operands[0])
        c1 = _get_constant_int(op.operands[1])
        if c1 is not None:
            val = symbolize_mlir_index(op.operands[0], loop_var_map,
                                       n_addr_bits, n_input_bits, memo)
            return val.and_constant(c1)
        if c0 is not None:
            val = symbolize_mlir_index(op.operands[1], loop_var_map,
                                       n_addr_bits, n_input_bits, memo)
            return val.and_constant(c0)
        # AND of two symbolic values → non-linear
        return F2Symbol.opaque(n_addr_bits, n_input_bits)

    # arith.ori → OR (check non-overlapping)
    if op_name == "arith.ori":
        lhs = symbolize_mlir_index(op.operands[0], loop_var_map, n_addr_bits,
                                   n_input_bits, memo)
        rhs = symbolize_mlir_index(op.operands[1], loop_var_map, n_addr_bits,
                                   n_input_bits, memo)
        if lhs.is_nonoverlapping(rhs):
            return lhs.or_nonoverlapping(rhs)
        return F2Symbol.opaque(n_addr_bits, n_input_bits)

    # arith.addi → add (check non-overlapping for carry-free)
    if op_name == "arith.addi":
        lhs = symbolize_mlir_index(op.operands[0], loop_var_map, n_addr_bits,
                                   n_input_bits, memo)
        rhs = symbolize_mlir_index(op.operands[1], loop_var_map, n_addr_bits,
                                   n_input_bits, memo)
        if lhs.is_nonoverlapping(rhs):
            return lhs.or_nonoverlapping(rhs)
        return F2Symbol.opaque(n_addr_bits, n_input_bits)

    # arith.muli → multiply by constant power of 2
    if op_name == "arith.muli":
        c0 = _get_constant_int(op.operands[0])
        c1 = _get_constant_int(op.operands[1])
        if c1 is not None and _is_power_of_2(c1):
            val = symbolize_mlir_index(op.operands[0], loop_var_map,
                                       n_addr_bits, n_input_bits, memo)
            return val.shift_left(int(math.log2(c1)))
        if c0 is not None and _is_power_of_2(c0):
            val = symbolize_mlir_index(op.operands[1], loop_var_map,
                                       n_addr_bits, n_input_bits, memo)
            return val.shift_left(int(math.log2(c0)))
        return F2Symbol.opaque(n_addr_bits, n_input_bits)

    # arith.index_cast / arith.extui / arith.extsi / arith.trunci → recurse
    if op_name in ("arith.index_cast", "arith.extui", "arith.extsi",
                   "arith.trunci"):
        return symbolize_mlir_index(op.operands[0], loop_var_map, n_addr_bits,
                                   n_input_bits, memo)

    # affine.load from scalar memref → trace through to the store
    # Allo lowers variable assignments (e.g., `il: uint32 = expr`) into
    # alloc(memref<i32>) + affine.store(expr) + affine.load, so the value
    # flows through a scalar memref. We find the store and symbolize its value.
    if op_name == "affine.load":
        memref_val = op.operands[0]
        stored_val = _find_scalar_store_value(memref_val, op)
        if stored_val is not None:
            return symbolize_mlir_index(stored_val, loop_var_map, n_addr_bits,
                                       n_input_bits, memo)
        return F2Symbol.opaque(n_addr_bits, n_input_bits)

    # memref.load from scalar memref → same pattern
    if op_name == "memref.load" and len(list(op.operands)) == 1:
        memref_val = op.operands[0]
        stored_val = _find_scalar_store_value(memref_val, op)
        if stored_val is not None:
            return symbolize_mlir_index(stored_val, loop_var_map, n_addr_bits,
                                       n_input_bits, memo)
        return F2Symbol.opaque(n_addr_bits, n_input_bits)

    # Unknown op → conservative opaque
    return F2Symbol.opaque(n_addr_bits, n_input_bits)


# ---------------------------------------------------------------------------
# Buffer conflict analysis
# ---------------------------------------------------------------------------


def _walk_ops_recursive(operations):
    """Yield all operations recursively, descending into nested regions.

    Uses the generic Operation API (op.regions) which works for all op types
    including affine.for, scf.for, scf.if, etc. without needing opview casts.
    """
    for op in operations:
        yield op
        # Generically descend into all regions/blocks of this op
        try:
            for region in op.regions:
                for block in region:
                    yield from _walk_ops_recursive(block.operations)
        except Exception:
            pass


def _find_enclosing_for_ops(op, func_op):
    """Find all enclosing for-loop ops for a given operation.

    Walks from the operation upward through parent operations to find
    enclosing scf.for or affine.for operations.

    In Allo's MLIR Python bindings, op.parent returns the parent Operation
    directly (not a Block).  The induction variable is accessed via
    op.regions[0].blocks[0].arguments[0] (generic Operation API).

    Returns:
        list of (for_op, induction_var) tuples, outermost first.
    """
    enclosing = []
    current = op
    while current is not None:
        try:
            parent_op = current.parent
            if parent_op is None:
                break
        except Exception:
            break
        try:
            parent_name = parent_op.name
        except Exception:
            break
        if parent_name in ("scf.for", "affine.for"):
            # Access induction variable via generic Operation region API
            try:
                block = list(list(parent_op.regions)[0])[0]
                iv = list(block.arguments)[0]
                enclosing.append((parent_op, iv))
            except Exception:
                pass
        elif parent_name == "func.func":
            break
        current = parent_op
    enclosing.reverse()  # outermost first
    return enclosing


def _get_loop_bound(for_op):
    """Extract the trip count of a for loop (assuming constant bounds 0..N).

    Returns the trip count as int, or None if bounds are not constant.
    """
    import re as _re

    op_name = for_op.name
    if op_name == "scf.for":
        lb = _get_constant_int(for_op.operands[0])
        ub = _get_constant_int(for_op.operands[1])
        step = _get_constant_int(for_op.operands[2])
        if lb is not None and ub is not None and step is not None and step > 0:
            return (ub - lb + step - 1) // step
    elif op_name == "affine.for":
        try:
            from .._mlir.ir import AffineMapAttr

            # Affine for uses lowerBoundMap and upperBoundMap attributes
            # which are affine maps like () -> (0) and () -> (8)
            lb_map_str = str(AffineMapAttr(for_op.attributes["lowerBoundMap"]).value)
            ub_map_str = str(AffineMapAttr(for_op.attributes["upperBoundMap"]).value)

            lb_match = _re.search(r"\(\)\s*->\s*\((\d+)\)", lb_map_str)
            ub_match = _re.search(r"\(\)\s*->\s*\((\d+)\)", ub_map_str)

            if lb_match and ub_match:
                lb = int(lb_match.group(1))
                ub = int(ub_match.group(1))

                # Get step from IntegerAttr
                from .._mlir.ir import IntegerAttr as MlirIntegerAttr
                step = MlirIntegerAttr(for_op.attributes["step"]).value
                if step > 0:
                    return (ub - lb + step - 1) // step
        except Exception:
            pass
    return None


def _is_loop_unrolled(for_op):
    """Check if a for loop has an unroll attribute."""
    try:
        attrs = for_op.attributes
        for attr_name in ("unroll", "pipeline"):
            if attr_name in attrs:
                return True
    except Exception:
        pass
    return False


def _get_memref_size(alloc_op):
    """Get the total size of a 1D memref from an alloc op."""
    try:
        from .._mlir.ir import MemRefType
        mtype = MemRefType(alloc_op.result.type)
        shape = mtype.shape
        if len(shape) == 1:
            return shape[0]
    except Exception:
        pass
    return None


def _get_alloc_name(alloc_op):
    """Get the name attribute of an alloc op."""
    try:
        from .._mlir.ir import StringAttr
        if "name" in alloc_op.attributes:
            return StringAttr(alloc_op.attributes["name"]).value
    except Exception:
        pass
    return None


def analyze_buffer_conflicts(func_op, alloc_op, n_addr_bits):
    """Analyze all load/store accesses on a 1D buffer and build the conflict subspace.

    Finds all load/store operations referencing the given alloc, identifies
    enclosing loop structure, determines which loops are unrolled, builds
    F2Symbol representations, and computes the conflict subspace P.

    Args:
        func_op: MLIR FuncOp containing the buffer.
        alloc_op: memref.alloc operation for the 1D buffer.
        n_addr_bits: number of address bits (log2 of array size).

    Returns:
        (P, parallel_var_bits, loop_info) where:
            P: numpy uint8 array — basis of conflict subspace (columns).
            parallel_var_bits: list of input bit indices for unrolled variables.
            loop_info: dict with loop metadata for diagnostics.
    """
    alloc_result = alloc_op.result

    # Collect memref.load/memref.store ops referencing this alloc.
    # We focus on memref.load/store (COMPUTE phase with bitwise index
    # expressions that are F2-linear) and skip affine.load/affine.store
    # (LOAD/WRITE phases with sequential access patterns that use affine
    # maps and would produce opaque symbols).
    access_ops = []
    for op in _walk_ops_recursive(func_op.entry_block.operations):
        op_name = op.name
        if op_name == "memref.load":
            if op.operands[0] == alloc_result:
                # Check this is an indexed load (has index operand)
                operands = list(op.operands)
                if len(operands) >= 2:
                    access_ops.append(("load", op, operands[1]))
        elif op_name == "memref.store":
            if op.operands[1] == alloc_result:
                operands = list(op.operands)
                if len(operands) >= 3:
                    access_ops.append(("store", op, operands[2]))

    if not access_ops:
        return (np.zeros((n_addr_bits, 0), dtype=np.uint8), [], {})

    # Collect all enclosing loops from all accesses, and track innermost loops.
    # IMPORTANT: done in a single pass because op.parent creates new Python
    # wrappers each time, so id() is only stable within a single call.
    all_loops = {}  # id(for_op) → (for_op, iv, trip_count, is_unrolled)
    innermost_loop_ids = set()
    for _, acc_op, _ in access_ops:
        enclosing = _find_enclosing_for_ops(acc_op, func_op)
        for for_op, iv in enclosing:
            fid = id(for_op)
            if fid not in all_loops:
                tc = _get_loop_bound(for_op)
                unrolled = _is_loop_unrolled(for_op)
                all_loops[fid] = (for_op, iv, tc, unrolled)
        if enclosing:
            innermost_loop_ids.add(id(enclosing[-1][0]))

    # Build loop_var_map: assign bit positions to loop induction variables
    loop_var_map = {}
    parallel_var_bits = []
    current_bit = 0
    loop_info = {}

    for fid, (for_op, iv, trip_count, unrolled) in all_loops.items():
        if trip_count is None or trip_count <= 0:
            continue
        n_bits = max(1, int(math.ceil(math.log2(max(trip_count, 2)))))
        loop_var_map[iv] = (current_bit, n_bits)
        # Mark as parallel if: has unroll/pipeline attribute, OR is the
        # innermost loop of an access nest (standard HLS unroll target).
        is_parallel = unrolled or (fid in innermost_loop_ids)
        if is_parallel:
            parallel_var_bits.extend(range(current_bit, current_bit + n_bits))
        loop_info[fid] = {
            "trip_count": trip_count,
            "n_bits": n_bits,
            "start_bit": current_bit,
            "unrolled": is_parallel,
        }
        current_bit += n_bits

    n_input_bits = max(current_bit, 1)

    # Symbolize all index expressions
    symbols = []
    memo = {}
    for acc_type, acc_op, idx_ssa in access_ops:
        if idx_ssa is None:
            # Affine access — treat as opaque
            symbols.append(F2Symbol.opaque(n_addr_bits, n_input_bits))
        else:
            sym = symbolize_mlir_index(idx_ssa, loop_var_map, n_addr_bits,
                                       n_input_bits, memo)
            symbols.append(sym)

    # Build conflict subspace
    P = build_conflict_subspace(symbols, parallel_var_bits)

    return (P, parallel_var_bits, loop_info)


# ---------------------------------------------------------------------------
# Top-level auto-F2 application
# ---------------------------------------------------------------------------


def _extract_stride_bits(P, n_addr_bits):
    """Extract stride bits from the conflict subspace basis.

    Examines basis vectors of P. Unit vectors (e_s) give stride_bit = s directly.
    Non-unit vectors are handled by the generalized solver.

    Args:
        P: numpy array of shape (n_addr_bits, rank) — conflict subspace basis.

    Returns:
        list of stride bit positions.
    """
    stride_bits = []
    if P.shape[1] == 0:
        return stride_bits

    for col_idx in range(P.shape[1]):
        vec = P[:, col_idx]
        nonzero = np.where(vec)[0]
        if len(nonzero) == 1:
            stride_bits.append(int(nonzero[0]))
        else:
            # Non-unit vector: pick the highest set bit as stride
            stride_bits.append(int(nonzero[-1]))

    return stride_bits


def _determine_banking_mode(P, n_addr_bits, bank_bits):
    """Determine whether cyclic or block banking is more appropriate.

    Cyclic banking works when the conflict subspace has stride bits
    within the bank bit range (bits below bank_bits). Block banking
    is used when conflicts are in the upper bits.

    Returns:
        ("cyclic", stride_bit) or ("block", None)
    """
    stride_bits = _extract_stride_bits(P, n_addr_bits)

    if not stride_bits:
        return ("cyclic", None)

    # Check if any stride bit is above the bank_bits range
    # This is the case for inter-stage FFT buffers
    has_high_stride = any(s >= bank_bits for s in stride_bits)
    has_low_stride = any(s < bank_bits for s in stride_bits)

    # Check if block banking is more appropriate.
    # Block banking: bank = addr >> offset_bits, offset = addr & (depth-1).
    # Works when all conflict bits are >= offset_bits (= n_addr_bits - bank_bits).
    offset_bits = n_addr_bits - bank_bits
    all_above_offset = all(s >= offset_bits for s in stride_bits)
    if all_above_offset and len(stride_bits) >= bank_bits:
        return ("block", None)

    if has_high_stride and not has_low_stride:
        # Conflicts only from high stride bits → cyclic with XOR swizzle
        high_strides = [s for s in stride_bits if s >= bank_bits]
        return ("cyclic", high_strides[0])

    if not has_high_stride and has_low_stride:
        # All conflicts in low bits → already separated by cyclic banking
        # No swizzle needed
        return ("cyclic", None)

    if has_high_stride and has_low_stride:
        # Mixed — try cyclic with highest stride
        high_strides = [s for s in stride_bits if s >= bank_bits]
        return ("cyclic", high_strides[0])

    return ("cyclic", None)


def auto_apply_f2(module, func_name, func_args, bank_bits=None,
                  kernel_names=None):
    """Analyze all 1D buffers in a function and apply F2 layouts automatically.

    Steps:
    1. Find the function in the module.
    2. Find all 1D memref.alloc operations.
    3. For each alloc, call analyze_buffer_conflicts.
    4. If the conflict subspace P is non-empty, determine banking parameters.
    5. Apply f2_layout transform with partition, bind_storage, dependence pragmas.

    Args:
        module: MLIR Module.
        func_name: name of the function to analyze.
        func_args: dict of function arguments (from Schedule).
        bank_bits: number of bank bits. If None, determined from unrolled loop bound.
        kernel_names: list of kernel function names to analyze, or None for all.
    """
    from .._mlir.ir import (
        StringAttr,
        MemRefType,
    )
    from .._mlir.dialects import (
        memref as memref_d,
        func as func_d,
    )
    from ..ir.utils import MockBuffer
    from ..ir.transform import find_buffer

    applied = []

    # Find all functions to analyze
    target_funcs = []
    for op in module.body.operations:
        if isinstance(op, func_d.FuncOp):
            fn = StringAttr(op.attributes["sym_name"]).value
            if kernel_names is None or fn in kernel_names:
                target_funcs.append((fn, op))

    for fn_name, func_op in target_funcs:
        # Skip if not in func_args
        if fn_name not in func_args:
            continue

        # Find all 1D alloc ops in this function
        allocs = []
        for op in _walk_ops_recursive(func_op.entry_block.operations):
            if isinstance(op, memref_d.AllocOp):
                try:
                    mtype = MemRefType(op.result.type)
                    shape = mtype.shape
                    if len(shape) == 1 and shape[0] > 1:
                        buf_name = _get_alloc_name(op)
                        if buf_name is not None:
                            allocs.append((buf_name, op, shape[0]))
                except Exception:
                    continue

        for buf_name, alloc_op, buf_size in allocs:
            if not _is_power_of_2(buf_size):
                continue

            # Skip buffers with uses that apply_f2_layout can't rewrite
            # (e.g., allo.stream_put/get, func.return, etc.).
            _handled_ops = {
                "memref.load", "memref.store",
                "affine.load", "affine.store",
                "allo.partition",
            }
            has_unhandled_use = False
            for use in alloc_op.result.uses:
                if use.owner.name not in _handled_ops:
                    has_unhandled_use = True
                    break
            if has_unhandled_use:
                continue

            n_addr_bits = int(math.log2(buf_size))
            P, parallel_var_bits, loop_info = analyze_buffer_conflicts(
                func_op, alloc_op, n_addr_bits
            )

            if P.shape[1] == 0:
                # No conflicts detected
                continue

            # Determine bank_bits from parallel variable dimensions
            effective_bank_bits = bank_bits
            if effective_bank_bits is None:
                # Use the conflict subspace dimension as minimum bank bits
                effective_bank_bits = P.shape[1]
                # Cap at n_addr_bits
                effective_bank_bits = min(effective_bank_bits, n_addr_bits)

            # Determine banking mode and stride
            banking_mode, stride_bit = _determine_banking_mode(
                P, n_addr_bits, effective_bank_bits
            )

            # Skip buffers where cyclic banking has no high stride bit.
            # These are cases where all conflicts are in the low bank_bits
            # (resolved by standard cyclic partition without XOR swizzle).
            if banking_mode == "cyclic" and stride_bit is None:
                continue

            # For block banking, stride_bit holds offset_bits = n_bits - bank_bits
            effective_stride = stride_bit
            if banking_mode == "block":
                effective_stride = n_addr_bits - effective_bank_bits

            try:
                apply_f2_layout(
                    module, fn_name, buf_name, func_args,
                    n_addr_bits, effective_bank_bits, effective_stride,
                    banking=banking_mode,
                )
                applied.append((fn_name, buf_name, banking_mode,
                                effective_bank_bits, effective_stride))
            except Exception as e:
                # If layout application fails, skip this buffer
                import warnings
                warnings.warn(
                    f"auto_f2: failed to apply F2 layout to "
                    f"{fn_name}:{buf_name}: {e}"
                )

    return applied
