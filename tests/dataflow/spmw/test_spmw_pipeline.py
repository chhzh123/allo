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
