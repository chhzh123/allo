# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo.spmw as spmw
from allo.ir.types import float32

M, N, K = 16, 16, 16


def _stripe_twin():
    grid = spmw.mesh((1, N))

    @spmw.unit
    def pe(ctx):
        for m in range(M):
            c: float32 = 0
            for _ in range(K):
                a: float32 = ctx.west.get()
                b: float32 = ctx.north.get()
                c += a * b
                ctx.east.put(a)
            ctx.c_local[m] = c

    @spmw.region()
    def top(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    return top


def test_1d_systolic_twin_matches_dataflow():
    # a 1-D output-stationary systolic stripe: A broadcasts across the columns, B is fed per column,
    # each column computes one output column -- matching the hand-written df 1D-systolic original
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    spmw.build(_stripe_twin(), target="simulator")(A, B, C)
    np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
