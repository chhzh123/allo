# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
import allo.spmw as spmw
from allo.ir.types import float32, index

M, N = 4, 4


def index_calculation(x: index) -> index:
    res: index = x - 1
    return res


def _func_index_twin():
    @spmw.unit
    def producer(ctx):
        for i, j in allo.grid(M, N):
            out: float32 = ctx.A[i, index_calculation(j) + 1]
            ctx.pipe.put(out)

    @spmw.unit
    def consumer(ctx):
        for i, j in allo.grid(M, N):
            data: float32 = ctx.pipe.get()
            ctx.B[i, j] = data + 1

    @spmw.region()
    def top(A: float32[M, N], B: float32[M, N]):
        spmw.map(producer, grid=(1,))
        spmw.map(consumer, grid=(1,))
        spmw.channel("pipe", float32, depth=4)

    return top


def test_func_index_twin_matches_dataflow():
    # the producer's index uses a module-level helper (index_calculation(j) + 1 == j); the desugar
    # captures the helper into the generated module. Matches the df original (B == A + 1).
    A = np.random.rand(M, N).astype(np.float32)
    B = np.zeros((M, N), dtype=np.float32)
    spmw.build(_func_index_twin(), target="simulator")(A, B)
    np.testing.assert_array_equal(B, A + 1)
