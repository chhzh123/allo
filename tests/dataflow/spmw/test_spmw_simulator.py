# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo.spmw as spmw
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


def test_systolic_twin_runs_on_simulator():
    M, N, K = 2, 2, 2
    module = spmw.build(_systolic_twin(M, N, K), target="simulator")
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    module(A, B, C)
    np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)


def test_non_systolic_region_rejected():
    grid = spmw.mesh((3, 3))

    @spmw.unit
    def pe(ctx):
        ctx.west.get()

    @spmw.region()
    def r(A: float32[3, 3]):
        spmw.map(pe, grid=grid)  # no stream_in flows -> not the systolic pattern

    with pytest.raises(NotImplementedError):
        spmw.build(r, target="simulator")
