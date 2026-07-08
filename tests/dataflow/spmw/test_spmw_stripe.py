# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
import allo.dataflow as df
import allo.spmw as spmw
from allo.ir.types import float32, Stream

M, N, K = 16, 16, 16
P0 = K + 2


@df.region()
def _df_original(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    fifo_A: Stream[float32, 4][2, P0]
    fifo_B: Stream[float32, 4][2, P0]

    @df.kernel(mapping=[2, P0], args=[A, B, C])
    def gemm(local_A: float32[M, K], local_B: float32[K, N], local_C: float32[M, N]):
        i, j = df.get_pid()
        with allo.meta_if(i == 0 and (j == 0 or j == P0 - 1)):
            pass
        with allo.meta_elif(i == 0):
            for _ in range(N):
                for k in range(K):
                    fifo_B[i + 1, j].put(local_B[k, j - 1])
        with allo.meta_elif(j == 0):
            for m in range(M):
                for k in range(K):
                    fifo_A[i, j + 1].put(local_A[m, k])
        with allo.meta_elif(j == P0 - 1):
            for m in range(M):
                for _ in range(K):
                    a: float32 = fifo_A[i, j].get()
        with allo.meta_else():
            for m in range(M):
                c: float32 = 0
                for _ in range(K):
                    a: float32 = fifo_A[i, j].get()
                    b: float32 = fifo_B[i, j].get()
                    c += a * b
                    fifo_A[i, j + 1].put(a)
                local_C[m, j - 1] = c


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
    # the concise SPMW stripe twin and the hand-written df 1D-systolic original run bit-identically
    # on the same input (same float accumulation order), and both compute A @ B
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    c_df = np.zeros((M, N), dtype=np.float32)
    c_spmw = np.zeros((M, N), dtype=np.float32)
    df.build(_df_original, target="simulator")(A, B, c_df)
    spmw.build(_stripe_twin(), target="simulator")(A, B, c_spmw)
    np.testing.assert_array_equal(c_spmw, c_df)  # bit-identical to the df original
    np.testing.assert_allclose(c_spmw, np.dot(A, B), atol=1e-5)


def test_rectangular_stripe_fails_closed():
    # the copied stripe template only balances for the square M == N == K case; a rectangular shape
    # is rejected rather than silently miscomputing
    import pytest

    RM, RN, RK = 8, 16, 16
    grid = spmw.mesh((1, RN))

    @spmw.unit
    def pe(ctx):
        for m in range(RM):
            c: float32 = 0
            for _ in range(RK):
                a: float32 = ctx.west.get()
                b: float32 = ctx.north.get()
                c += a * b
                ctx.east.put(a)
            ctx.c_local[m] = c

    @spmw.region()
    def top(A: float32[RM, RK], B: float32[RK, RN], C: float32[RM, RN]):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    with pytest.raises(Exception, match="square M == N == K"):
        spmw.build(top, target="simulator")
