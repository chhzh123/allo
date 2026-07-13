# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""task6.3 -- SPMW L1 twin of ``tests/dataflow/sparse/test_simple_sparse_systolic.py``.

The sparse systolic GEMM (`semm`) is the output-stationary systolic array with a `if a != 0` guard
before the multiply-accumulate. The SPMW twin expresses it as one work unit over a mesh with the same
guard; the SPMW systolic desugar transcribes the interior body verbatim, so the guard carries through.
The original `allo.dataflow` kernel is kept here as the oracle and the SPMW simulator output is asserted
bit-identical to it (and to ``numpy``) on the same sparse inputs (AC-3 / AC-9).
"""

import numpy as np
import allo
import allo.dataflow as df
import allo.spmw as spmw
from allo.ir.types import int32, Stream

M, N, K = 4, 4, 4
P0, P1 = M + 2, N + 2


# ---- the df original, kept verbatim as the L1 oracle ------------------------------------------------
@df.region()
def _df_sparse_top(A: int32[M, K], B: int32[K, N], C: int32[M, N]):
    fifo_A: Stream[int32, 4][P0, P1]
    fifo_B: Stream[int32, 4][P0, P1]

    @df.kernel(mapping=[P0, P1], args=[A, B, C])
    def semm(local_A: int32[M, K], local_B: int32[K, N], local_C: int32[M, N]):
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
                b: int32 = fifo_B[i, j].get()
        with allo.meta_elif(j == N + 1 and i > 0):
            for k in range(K):
                a: int32 = fifo_A[i, j].get()
        with allo.meta_else():
            c: int32 = 0
            for k in range(K):
                a: int32 = fifo_A[i, j].get()
                b: int32 = fifo_B[i, j].get()
                if a != 0:
                    c += a * b
                fifo_A[i, j + 1].put(a)
                fifo_B[i + 1, j].put(b)
            local_C[i - 1, j - 1] = c


# ---- the SPMW twin: one unit over a mesh, same sparse-MAC guard -------------------------------------
def _spmw_sparse_twin():
    grid = spmw.mesh((M, N))

    @spmw.unit
    def semm(ctx):
        c: int32 = 0
        for k in range(K):
            a: int32 = ctx.west.get()
            b: int32 = ctx.north.get()
            if a != 0:
                c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def semm_region(A: int32[M, K], B: int32[K, N], C: int32[M, N]):
        spmw.map(semm, grid=grid)
        spmw.stream_in(A, into=semm, flow="W->E")
        spmw.stream_in(B, into=semm, flow="N->S")
        spmw.stream_out(C, from_=semm, where="local", as_="c_local")

    return semm_region


def _sparse_A(seed):
    """A dense-shaped A with 2:4 structured sparsity (two of every four entries zeroed)."""
    rng = np.random.default_rng(seed)
    A = rng.integers(1, 6, size=(M, K)).astype(np.int32)
    flat = A.reshape(-1)
    for block in range(0, flat.size, 4):
        idx = block + rng.choice(4, size=2, replace=False)
        flat[idx] = 0
    return flat.reshape(M, K)


def test_spmw_sparse_twin_matches_df_and_numpy():
    """The SPMW sparse-systolic twin is bit-identical to the df oracle and to numpy across seeds."""
    df_mod = df.build(_df_sparse_top, target="simulator")
    spmw_mod = spmw.build(_spmw_sparse_twin(), target="simulator")
    for seed in range(4):
        A = _sparse_A(seed)
        B = (
            np.random.default_rng(seed + 100)
            .integers(1, 6, size=(K, N))
            .astype(np.int32)
        )

        C_df = np.zeros((M, N), dtype=np.int32)
        df_mod(A, B, C_df)

        C_spmw = np.zeros((M, N), dtype=np.int32)
        spmw_mod(A, B, C_spmw)

        np.testing.assert_array_equal(
            C_spmw, C_df
        )  # twin is bit-identical to the df oracle
        np.testing.assert_array_equal(C_spmw, A @ B)  # and correct vs numpy
