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


def test_channel_without_consumer_rejected():
    @spmw.unit
    def producer(ctx):
        for i in range(4):
            ctx.pipe.put(i)

    @spmw.region()
    def top(A: float32[M, N]):
        spmw.map(producer, grid=(1,))
        spmw.channel("pipe", float32, depth=4)

    with pytest.raises(spmw.SPMWError, match="one producer map and one consumer map"):
        spmw.build(top, target="simulator")


def test_channel_alias_form_fails_closed():
    # a channel used through a ctx alias (h = ctx.pipe; h.put) is not lowerable -> fail closed
    @spmw.unit
    def producer(ctx):
        h = ctx.pipe
        for i in range(4):
            h.put(i)

    @spmw.unit
    def consumer(ctx):
        for i in range(4):
            x = ctx.pipe.get()

    @spmw.region()
    def top(A: float32[M, N]):
        spmw.map(producer, grid=(1,))
        spmw.map(consumer, grid=(1,))
        spmw.channel("pipe", float32, depth=4)

    with pytest.raises(Exception):
        spmw.build(top, target="simulator")


def test_multi_dim_replication_rejected():
    # a multi-D replicated grid would give a tuple pid, which the scalar channel indexing does not
    # handle -- the pipeline path rejects it (multi-D systolic arrays are the mesh family)
    @spmw.unit
    def producer(ctx):
        pi = ctx.rank()
        ctx.pipe.put(1)

    @spmw.unit
    def consumer(ctx):
        pi = ctx.rank()
        x = ctx.pipe.get()

    @spmw.region()
    def top(A: float32[M, N]):
        spmw.map(producer, grid=(2, 2))
        spmw.map(consumer, grid=(2, 2))
        spmw.channel("pipe", float32, depth=4)

    with pytest.raises(Exception, match="1-D replication only"):
        spmw.build(top, target="simulator")


def test_singleton_rank_is_zero():
    # a size-1 unit's ctx.rank() resolves to the constant 0 (not left as an undefined ctx.rank())
    @spmw.unit
    def producer(ctx):
        for i, j in allo.grid(M, N):
            out: float32 = ctx.A[i, j]
            ctx.pipe.put(out)

    @spmw.unit
    def consumer(ctx):
        pi = ctx.rank()
        for i, j in allo.grid(M, N):
            data: float32 = ctx.pipe.get()
            ctx.B[i, j] = data + pi  # pi == 0, so B == A

    @spmw.region()
    def top(A: float32[M, N], B: float32[M, N]):
        spmw.map(producer, grid=(1,))
        spmw.map(consumer, grid=(1,))
        spmw.channel("pipe", float32, depth=4)

    A = np.random.rand(M, N).astype(np.float32)
    B = np.zeros((M, N), dtype=np.float32)
    spmw.build(top, target="simulator")(A, B)
    np.testing.assert_array_equal(B, A)


def test_channel_self_loop_rejected():
    # one map that both puts and gets the same channel is a self-loop, not a producer/consumer pair
    @spmw.unit
    def loopback(ctx):
        for i in range(4):
            ctx.pipe.put(i)
            x = ctx.pipe.get()

    @spmw.region()
    def top(A: float32[M, N]):
        spmw.map(loopback, grid=(1,))
        spmw.channel("pipe", float32, depth=4)

    with pytest.raises(spmw.SPMWError, match="self-loop"):
        spmw.build(top, target="simulator")


def test_lower_carries_channels_in_rolled_ir():
    # a pipeline region's rolled spmw.map IR preserves its inter-unit channels as #spmw.channel attrs
    # on the top function, so they survive to codegen (an M2 pass resolves them)
    text = str(spmw.lower(_producer_consumer_twin()))
    assert "spmw.channels" in text
    assert '#spmw.channel<name = "pipe"' in text
    assert "depth = 4" in text
    # two units -> two maps, and the channel is carried once at the top
    assert text.count("spmw.map") == 2
