# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile

import numpy as np
import pytest
import allo.spmw as spmw
import allo.backend.hls as hls
from allo.ir.types import float32


def _systolic_twin(M, N, K):
    grid = spmw.mesh((M, N))

    @spmw.unit
    def pe(ctx):
        c: float32 = 0
        for k in range(K):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    return gemm


def test_systolic_twin_emits_vitis_hls_code():
    # the desugared region produces synthesizable HLS C++ (no toolchain needed to inspect it)
    with tempfile.TemporaryDirectory() as tmp:
        module = spmw.build(_systolic_twin(2, 2, 2), target="vitis_hls", project=tmp)
    assert "void" in module.hls_code
    assert "hls::stream" in module.hls_code


@pytest.mark.skipif(
    not hls.is_available("vitis_hls"), reason="requires the Vitis HLS toolchain"
)
def test_systolic_twin_vitis_hls_csim():
    M, N, K = 2, 2, 2
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        module = spmw.build(
            _systolic_twin(M, N, K), target="vitis_hls", mode="csim", project=tmp
        )
        module(A, B, C)
    np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
