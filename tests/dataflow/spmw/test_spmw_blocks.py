# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
import allo.spmw as spmw
from allo.ir.types import int16

M, N = 4, 4
NUM_BLOCKS = 2


def _stream_of_blocks_twin():
    @spmw.unit
    def producer(ctx):
        for i in range(NUM_BLOCKS):
            block: int16[M, N] = 0
            for m in range(M):
                for n in range(N):
                    block[m, n] = ctx.A[i * M + m, n]
            ctx.pipe.put(block)

    @spmw.unit
    def consumer(ctx):
        for i in range(NUM_BLOCKS):
            block: int16[M, N] = ctx.pipe.get()
            for m in range(M):
                for n in range(N):
                    ctx.B[i * M + m, n] = block[m, n]

    @spmw.region()
    def top(A: int16[M * NUM_BLOCKS, N], B: int16[M * NUM_BLOCKS, N]):
        spmw.map(producer, grid=(1,))
        spmw.map(consumer, grid=(1,))
        spmw.channel("pipe", int16[M, N], depth=4)

    return top


def test_stream_of_blocks_twin_matches_dataflow():
    # a stream whose payload is a 2-D block: the producer packs A into blocks and streams them, the
    # consumer unpacks them into B -- matching the df original (B == A)
    A = np.random.randint(0, 100, (M * NUM_BLOCKS, N), dtype=np.int16)
    B = np.zeros((M * NUM_BLOCKS, N), dtype=np.int16)
    spmw.build(_stream_of_blocks_twin(), target="simulator")(A, B)
    np.testing.assert_array_equal(B, A)
