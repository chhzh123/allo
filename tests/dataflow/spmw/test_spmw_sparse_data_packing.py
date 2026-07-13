# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""task6.3 -- SPMW L1 twin of ``tests/dataflow/sparse/test_sparse_systolic_data_packing.py``.

The data-packing sparse GEMM takes A compressed as (nonzero values `A_nz`, column indices `A_in`) and
packs each B column into an ``int128`` streamed down the array; each PE reads the packed column once,
then for every nonzero reads a value + index, bit-slices the matching B entry out of the packed word,
multiply-accumulates, and forwards value/index horizontally and the packed column vertically.

The SPMW twin expresses the compute unit (with a third `idx` horizontal lane and the `int128` bit-slice
extraction) and lets the SPMW sparse-packed desugar generate the compressed-A loader, the real
``int128`` B-column packer, and the drains -- the pack runs in the generated dataflow boundary path,
not as a numpy shortcut. The `df` original is kept here as the L1 oracle and the SPMW simulator output
is asserted bit-identical to it (and to numpy) on the same compressed-sparse inputs (AC-3 / AC-9).
"""

import numpy as np
import allo
import allo.dataflow as df
import allo.spmw as spmw
from allo.ir.types import int32, int128, index, Stream

M, N, K = 4, 4, 4
P0, P1 = M + 2, N + 2
NZ = K // 2


# ---- the df original, kept verbatim as the L1 oracle ------------------------------------------------
@df.region()
def _df_packed_top(
    A_nz: int32[M, NZ], A_in: int32[M, NZ], B: int32[K, N], C: int32[M, N]
):
    fifo_A: Stream[int32, 4][P0, P1]
    fifo_idx: Stream[int32, 4][P0, P1]
    fifo_B: Stream[int128, 4][P0, P1]

    @df.kernel(mapping=[P0, P1], args=[A_nz, A_in, B, C])
    def semm(
        local_A_nz: int32[M, NZ],
        local_A_in: int32[M, NZ],
        local_B: int32[K, N],
        local_C: int32[M, N],
    ):
        i, j = df.get_pid()
        with allo.meta_if(i in {0, M + 1} and j in {0, N + 1}):
            pass
        with allo.meta_elif(j == 0):
            for knz in range(NZ):
                fifo_A[i, j + 1].put(local_A_nz[i - 1, knz])
                fifo_idx[i, j + 1].put(local_A_in[i - 1, knz])
        with allo.meta_elif(i == 0):
            pack: int128 = 0
            for k in range(K):
                msb: index = (k + 1) * 32 - 1
                lsb: index = k * 32
                b: int32 = local_B[k, j - 1]
                pack[lsb:msb] = b
            fifo_B[i + 1, j].put(pack)
        with allo.meta_elif(i == M + 1 and j > 0):
            b: int128 = fifo_B[i, j].get()
        with allo.meta_elif(j == N + 1 and i > 0):
            for k in range(NZ):
                a: int32 = fifo_A[i, j].get()
                idx: int32 = fifo_idx[i, j].get()
        with allo.meta_else():
            c: int32 = 0
            b_packed: int128 = fifo_B[i, j].get()
            for k in range(NZ):
                a: int32 = fifo_A[i, j].get()
                idx: int32 = fifo_idx[i, j].get()
                msb: index = (idx + 1) * 32 - 1
                lsb: index = idx * 32
                b: int32 = b_packed[lsb:msb]
                c += a * b
                fifo_A[i, j + 1].put(a)
                fifo_idx[i, j + 1].put(idx)
            fifo_B[i + 1, j].put(b_packed)
            local_C[i - 1, j - 1] = c


# ---- the SPMW twin: the compute unit; the desugar generates the packing peripherals ----------------
def _spmw_packed_twin():
    grid = spmw.mesh((M, N))

    @spmw.unit
    def semm(ctx):
        c: int32 = 0
        b_packed: int128 = ctx.north.get()
        for k in range(NZ):
            a: int32 = ctx.west.get()
            idx: int32 = ctx.idx.get()
            msb: index = (idx + 1) * 32 - 1
            lsb: index = idx * 32
            b: int32 = b_packed[lsb:msb]
            c += a * b
            ctx.east.put(a)
            ctx.idx.put(idx)
        ctx.south.put(b_packed)
        ctx.c_local[0] = c

    @spmw.region()
    def semm_region(
        A_nz: int32[M, NZ], A_in: int32[M, NZ], B: int32[K, N], C: int32[M, N]
    ):
        spmw.map(semm, grid=grid)
        spmw.stream_in(A_nz, into=semm, flow="W->E")  # compressed A values, horizontal
        spmw.stream_in(
            A_in, into=semm, as_="idx"
        )  # compressed A column indices, horizontal
        spmw.stream_in(
            B, into=semm, flow="N->S"
        )  # B columns, packed int128 by the loader, vertical
        spmw.stream_out(C, from_=semm, where="local", as_="c_local")

    return semm_region


def _make_sparse(seed):
    """A dense [M,K] matrix with NZ nonzeros per row, plus its compressed (values, indices) form."""
    rng = np.random.default_rng(seed)
    A = np.zeros((M, K), dtype=np.int32)
    A_nz = np.zeros((M, NZ), dtype=np.int32)
    A_in = np.zeros((M, NZ), dtype=np.int32)
    for i in range(M):
        cols = rng.choice(K, size=NZ, replace=False)
        vals = rng.integers(1, 10, size=NZ).astype(np.int32)
        for t, (col, val) in enumerate(zip(cols, vals)):
            A[i, col] = val
            A_nz[i, t] = val
            A_in[i, t] = col
    return A, A_nz, A_in


def test_spmw_packed_twin_matches_df_and_numpy():
    """The SPMW data-packing twin is bit-identical to the df oracle and to numpy across seeds."""
    df_mod = df.build(_df_packed_top, target="simulator")
    spmw_mod = spmw.build(_spmw_packed_twin(), target="simulator")
    for seed in range(4):
        A, A_nz, A_in = _make_sparse(seed)
        B = (
            np.random.default_rng(seed + 100)
            .integers(1, 10, size=(K, N))
            .astype(np.int32)
        )

        C_df = np.zeros((M, N), dtype=np.int32)
        df_mod(A_nz, A_in, B, C_df)

        C_spmw = np.zeros((M, N), dtype=np.int32)
        spmw_mod(A_nz, A_in, B, C_spmw)

        np.testing.assert_array_equal(
            C_spmw, C_df
        )  # twin is bit-identical to the df oracle
        np.testing.assert_array_equal(C_spmw, A @ B)  # and correct vs numpy
