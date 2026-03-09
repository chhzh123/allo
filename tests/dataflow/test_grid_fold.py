# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for grid folding (@df.kernel fold={} parameter).

Grid folding reduces PE count by converting spatial PEs into temporal
(unrolled) loops.  For example, mapping=[8] fold={0: 4} creates 2 PEs
instead of 8, each with an unrolled loop of 4 iterations.
"""

import pytest
import allo
from allo.ir.types import float32, int32, ConstExpr
import allo.dataflow as df
import numpy as np


def test_fold_basic_compile():
    """Test that a folded kernel compiles to MLIR without errors."""
    M = 8
    FOLD = 4

    @df.region()
    def top(A: float32[M], B: float32[M]):
        @df.kernel(mapping=[M], fold={0: FOLD}, args=[A, B])
        def compute(local_A: float32[M], local_B: float32[M]):
            idx = df.get_pid()
            local_B[idx] = local_A[idx] + 1.0

    s = df.customize(top)
    mlir_str = str(s.module)
    # Verify reduced PE count: 8/4 = 2 kernel functions
    assert "compute_0" in mlir_str
    assert "compute_1" in mlir_str
    # Verify fold loop with unroll annotation
    assert "unroll" in mlir_str
    print("Grid fold basic compile PASSED!")


def test_fold_2d_compile():
    """Test grid folding on one dimension of a 2D mapping."""
    M, N = 4, 8
    FOLD_N = 4

    @df.region()
    def top(A: float32[M, N], B: float32[M, N]):
        @df.kernel(mapping=[M, N], fold={1: FOLD_N}, args=[A, B])
        def compute(local_A: float32[M, N], local_B: float32[M, N]):
            i, j = df.get_pid()
            local_B[i, j] = local_A[i, j] + 1.0

    s = df.customize(top)
    mlir_str = str(s.module)
    # 4 * (8/4) = 8 kernel functions
    assert "compute_0_0" in mlir_str
    assert "compute_3_1" in mlir_str
    # No compute_X_2 should exist (only 0..1 for folded dim)
    assert "compute_0_2" not in mlir_str
    print("Grid fold 2D compile PASSED!")


def test_fold_with_bitwise_ops():
    """Test folded kernel with bitwise index expressions (FFT-like pattern)."""
    N = 16
    FOLD = 4

    @df.region()
    def top(A: float32[N], B: float32[N]):
        @df.kernel(mapping=[N], fold={0: FOLD}, args=[A, B])
        def compute(local_A: float32[N], local_B: float32[N]):
            idx = df.get_pid()
            # XOR-based access pattern (like FFT butterfly)
            partner: int32 = idx ^ 1
            local_B[idx] = local_A[idx] + local_A[partner]

    s = df.customize(top)
    mlir_str = str(s.module)
    # Should have 16/4 = 4 kernel functions
    assert "compute_0" in mlir_str
    assert "compute_3" in mlir_str
    assert "compute_4" not in mlir_str
    # Verify XOR operation in output
    assert "arith.xori" in mlir_str
    print("Grid fold with bitwise ops PASSED!")


def test_fold_constexpr_guard():
    """ConstExpr on a folded dimension should raise a clear error.

    Allo's customize() calls sys.exit(1) on compilation errors, so we
    catch SystemExit instead of RuntimeError.
    """

    def make_bad_kernel():
        N = 8
        FOLD = 4

        def helper(x):
            return x * 2

        @df.region()
        def top(A: float32[N], B: float32[N]):
            @df.kernel(mapping=[N], fold={0: FOLD}, args=[A, B])
            def compute(local_A: float32[N], local_B: float32[N]):
                idx = df.get_pid()
                # ConstExpr depends on folded pid — should fail
                doubled: ConstExpr[int32] = helper(idx)
                local_B[doubled] = local_A[idx]

        df.customize(top)

    with pytest.raises(SystemExit):
        make_bad_kernel()
    print("Grid fold ConstExpr guard PASSED!")


def test_fold_with_auto_f2():
    """End-to-end test: fold + auto_f2 for conflict-free partitioning.

    Uses an FFT-like stride-4 butterfly pattern where each folded iteration
    accesses buf[b] and buf[b ^ 4]. This creates a conflict subspace with
    stride bit 2 (beyond bank_bits), requiring XOR-swizzle partitioning.
    """
    N = 32
    FOLD = 8  # 32/8 = 4 PEs, each with 8-iter unrolled loop
    STRIDE = 4  # butterfly stride — XOR with 4

    @df.region()
    def top(
        inp_re: float32[N],
        out_re: float32[N],
    ):
        @df.kernel(
            mapping=[N],
            fold={0: FOLD},
            args=[inp_re, out_re],
        )
        def butterfly(
            local_in: float32[N],
            local_out: float32[N],
        ):
            b = df.get_pid()
            # Local buffer for intermediate computation
            buf: float32[N]
            buf[b] = local_in[b]
            # Stride-4 butterfly pair: b and b ^ 4
            partner: int32 = b ^ STRIDE
            a: float32 = buf[b]
            p: float32 = buf[partner]
            local_out[b] = a + p

    s = df.customize(top)
    mlir_str = str(s.module)

    # Verify fold: 4 PEs instead of 32
    assert "butterfly_0" in mlir_str
    assert "butterfly_3" in mlir_str
    assert "butterfly_4" not in mlir_str

    # Apply auto_f2 — should detect XOR stride-4 conflict and partition
    s.auto_f2()
    mlir_after = str(s.module)

    # auto_f2 should have transformed buf from 1D to 2D (8 banks x 4 depth)
    assert "8x4xf32" in mlir_after, (
        f"Expected 8x4xf32 memref in MLIR after auto_f2"
    )
    print("Grid fold + auto_f2 PASSED!")


if __name__ == "__main__":
    test_fold_basic_compile()
    test_fold_2d_compile()
    test_fold_with_bitwise_ops()
    test_fold_constexpr_guard()
    test_fold_with_auto_f2()
