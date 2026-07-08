# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo
import allo.spmw as spmw
from allo.ir.types import float32

M, N = 16, 16


def _pipeline_region():
    @spmw.unit
    def producer(ctx):
        for i in range(4):
            ctx.pipe.put(i)

    @spmw.unit
    def consumer(ctx):
        for i in range(4):
            x = ctx.pipe.get()

    @spmw.region()
    def top(A, B):
        spmw.map(producer, grid=(1,))
        spmw.map(consumer, grid=(1,))
        spmw.channel("pipe", float32, depth=4)

    return top


def test_channel_is_collected():
    coll = spmw._collect(_pipeline_region())
    assert [c.name for c in coll.channels] == ["pipe"]
    assert coll.channels[0].depth == 4
    # two different units mapped in one region -- a producer/consumer pipeline
    assert len(coll.maps) == 2


def test_channel_port_passes_body_validation():
    # a unit body may put/get on a declared channel as if it were a port
    coll = spmw._validate_collection(spmw._collect(_pipeline_region()))
    assert {c.name for c in coll.channels} == {"pipe"}


def test_undeclared_channel_port_rejected():
    @spmw.unit
    def producer(ctx):
        for i in range(4):
            ctx.pipe.put(i)  # "pipe" is never declared as a channel

    @spmw.region()
    def top(A):
        spmw.map(producer, grid=(1,))

    with pytest.raises(spmw.SPMWError, match="undeclared port"):
        spmw._validate_collection(spmw._collect(top))


def test_duplicate_channel_rejected():
    @spmw.unit
    def u(ctx):
        pass

    @spmw.region()
    def top(A):
        spmw.map(u, grid=(1,))
        spmw.channel("pipe", float32)
        spmw.channel("pipe", float32)

    with pytest.raises(spmw.SPMWError, match="declared more than once"):
        spmw._collect(top)


def _producer_consumer_twin():
    @spmw.unit
    def producer(ctx):
        for i, j in allo.grid(M, N):
            out: float32 = ctx.A[i, j]
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


def test_producer_consumer_twin_matches_dataflow():
    # the concise SPMW twin desugars to a producer/consumer dataflow pipeline and runs
    # bit-identically to the hand-written df original (which produces B = A + 1 exactly)
    A = np.random.rand(M, N).astype(np.float32)
    B = np.zeros((M, N), dtype=np.float32)
    spmw.build(_producer_consumer_twin(), target="simulator")(A, B)
    np.testing.assert_array_equal(B, A + 1)


P0 = 2
Mt = M // P0


def _cooperative_gemv_twin():
    @spmw.unit
    def gemv0(ctx):
        pi = ctx.rank()
        y_out: float32[Mt] = 0
        for m in range(pi * Mt, (pi + 1) * Mt):
            y_acc: float32 = 0
            for n in range(N // 2):
                y_acc += ctx.A[m, n] * ctx.x[n]
            y_out[m - pi * Mt] = y_acc
        ctx.pipe.put(y_out)

    @spmw.unit
    def gemv1(ctx):
        pi = ctx.rank()
        y_out: float32[Mt] = 0
        for m in range(pi * Mt, (pi + 1) * Mt):
            y_acc: float32 = 0
            for n in range(N // 2, N):
                y_acc += ctx.A[m, n] * ctx.x[n]
            y_out[m - pi * Mt] = y_acc
        y_prev: float32[Mt] = ctx.pipe.get()
        for m in range(pi * Mt, (pi + 1) * Mt):
            ctx.y[m] = y_out[m - pi * Mt] + y_prev[m - pi * Mt]

    @spmw.region()
    def top(A: float32[M, N], x: float32[N], y: float32[M]):
        spmw.map(gemv0, grid=(P0,))
        spmw.map(gemv1, grid=(P0,))
        spmw.channel("pipe", float32[Mt], depth=2)

    return top


def test_cooperative_gemv_twin_matches_dataflow():
    # a replicated two-stage GEMV: each of P0 PEs computes half the reduction, hands its partial
    # over a pid-indexed vector channel, and the second stage sums them -- matching the df original
    A = np.random.rand(M, N).astype(np.float32)
    x = np.random.rand(N).astype(np.float32)
    y = np.zeros((M,), dtype=np.float32)
    spmw.build(_cooperative_gemv_twin(), target="simulator")(A, x, y)
    np.testing.assert_allclose(y, np.dot(A, x), atol=1e-5)
