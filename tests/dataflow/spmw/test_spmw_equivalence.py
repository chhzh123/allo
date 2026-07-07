# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import numpy as np
import allo
from allo.ir.types import float32, Stream
import allo.dataflow as df
import allo.spmw as spmw

M, N, K = 3, 4, 5
P0, P1 = M + 2, N + 2


# --- the hand-written allo.dataflow original: a systolic GEMM spelled out with an explicit PID
# chain -- get_pid plus five meta_if/elif/else cases (loaders, drains, corners, interior) and
# explicit fifo[i, j+1] index arithmetic. This is the boilerplate SPMW is meant to replace. ---
@df.region()
def _df_original(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    fifo_A: Stream[float32, 4][P0, P1]
    fifo_B: Stream[float32, 4][P0, P1]

    @df.kernel(mapping=[P0, P1], args=[A, B, C])
    def gemm(local_A: float32[M, K], local_B: float32[K, N], local_C: float32[M, N]):
        i, j = df.get_pid()
        with allo.meta_if(i in {0, M + 1} and j in {0, N + 1}):
            pass
        with allo.meta_elif(j == 0):
            for k in range(K):
                fifo_A[i, j + 1].put(local_A[i - 1, k])
        with allo.meta_elif(i == 0):
            for k in range(K):
                fifo_B[i + 1, j].put(local_B[k, j - 1])
        with allo.meta_elif(i == M + 1 and j > 0):
            for k in range(K):
                b: float32 = fifo_B[i, j].get()
        with allo.meta_elif(j == N + 1 and i > 0):
            for k in range(K):
                a: float32 = fifo_A[i, j].get()
        with allo.meta_else():
            c: float32 = 0
            for k in range(K):
                a: float32 = fifo_A[i, j].get()
                b: float32 = fifo_B[i, j].get()
                c += a * b
                fifo_A[i, j + 1].put(a)
                fifo_B[i + 1, j].put(b)
            local_C[i - 1, j - 1] = c


# --- the SPMW twin: one work-unit body plus three declarative flows; boundary roles, FIFO
# families and PID routing are derived, not written. ---
def _spmw_twin():
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


def test_spmw_twin_is_bit_identical_to_dataflow_original():
    # concision without behaviour change: the concise twin and the hand-written df program must
    # produce the same result bit for bit (same float accumulation order), not merely close.
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    c_df = np.zeros((M, N), dtype=np.float32)
    c_spmw = np.zeros((M, N), dtype=np.float32)
    df.build(_df_original, target="simulator")(A, B, c_df)
    spmw.build(_spmw_twin(), target="simulator")(A, B, c_spmw)
    np.testing.assert_array_equal(c_spmw, c_df)


def test_spmw_twin_elides_the_pid_routing_boilerplate():
    # the df original hand-writes the PID dispatch the SPMW twin derives from the topology
    original = inspect.getsource(_df_original)
    twin = inspect.getsource(_spmw_twin)
    for boilerplate in ("get_pid", "meta_elif", "meta_else", "fifo_A[i, j + 1]"):
        assert boilerplate in original
        assert boilerplate not in twin

    # and it is materially shorter measured in non-blank, non-comment source lines
    def _loc(src):
        return sum(
            1
            for line in src.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    assert _loc(twin) < _loc(original)
