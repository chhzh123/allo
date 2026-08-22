# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fabric elaboration and the checks the design commits to.

Each negative case here is one row of the design's check table, and asserts on
the diagnostic rather than merely that something was raised -- a check that fires
with an unreadable message has only half done its job.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import float32, int8, int32

M = N = K = 4


class MacIO(spmw.Interface):
    west = spmw.In(float32)
    north = spmw.In(float32)
    east = spmw.Out(float32)
    south = spmw.Out(float32)
    c = spmw.MemOut(float32)


@spmw.unit
def pe(io: MacIO):
    acc: float32 = 0
    for k in range(K):
        a = io.west.get()
        b = io.north.get()
        acc += a * b
        io.east.put(a)
        io.south.put(b)
    io.c = acc


def test_gemm_graph():
    @spmw.fabric
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        P = spmw.place(pe, on=spmw.mesh(MacIO, (M, N)))
        spmw.stream_in(A, into=P.west, index=(P.rows, ...))
        spmw.stream_in(B, into=P.north, index=(..., P.cols))
        spmw.gather(C, from_=P.c)

    graph = spmw.elaborate(gemm)
    assert len(graph.placements) == 1
    assert [b.kind for b in graph.bindings] == ["stream_in", "stream_in", "gather"]
    # The `...` axis's extent is the token count per port -- time, not space.
    assert graph.bindings[0].extras["extent"] == K
    assert graph.bindings[1].extras["extent"] == K


def test_stream_in_requires_an_index():
    @spmw.fabric
    def bad(A: float32[M, K], C: float32[M, N]):
        P = spmw.place(pe, on=spmw.mesh(MacIO, (M, N)))
        spmw.stream_in(A, into=P.west)

    with pytest.raises(spmw.SPMWBindingError, match="index= is required"):
        spmw.elaborate(bad)


def test_index_is_bounds_checked_over_the_whole_domain():
    @spmw.fabric
    def bad(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        P = spmw.place(pe, on=spmw.mesh(MacIO, (M, N)))
        spmw.stream_in(A, into=P.west, index=(P.rows + M, ...))
        spmw.stream_in(B, into=P.north, index=(..., P.cols))
        spmw.gather(C, from_=P.c)

    with pytest.raises(spmw.SPMWBindingError, match="out of bounds"):
        spmw.elaborate(bad)


def test_an_unbound_in_must_be_covered():
    @spmw.fabric
    def bad(A: float32[M, K], C: float32[M, N]):
        P = spmw.place(pe, on=spmw.mesh(MacIO, (M, N)))
        spmw.stream_in(A, into=P.west, index=(P.rows, ...))
        spmw.gather(C, from_=P.c)

    with pytest.raises(spmw.SPMWUnboundError, match="north"):
        spmw.elaborate(bad)


def test_every_memin_must_be_bound():
    class WIO(spmw.Interface):
        a_in = spmw.In(int8)
        a_out = spmw.Out(int8)
        w = spmw.MemIn(int8)

    @spmw.unit
    def cell(io: WIO):
        io.a_out.put(io.a_in.get() * io.w)

    @spmw.fabric
    def bad(A: int8[4, 4]):
        P = spmw.place(
            cell,
            on=spmw.Topology(
                WIO,
                (4, 4),
                link=lambda i, j: {WIO.a_out: spmw.to((i, j + 1), WIO.a_in)},
            ),
        )
        spmw.stream_in(A, into=P.a_in, index=(..., P.rows))

    with pytest.raises(spmw.SPMWMemoryError, match="nothing supplies it"):
        spmw.elaborate(bad)


def test_a_destination_bound_whole_must_be_covered_whole():
    @spmw.fabric
    def bad(A: float32[M, K], B: float32[K, N], O: float32[M, 2 * N]):
        P = spmw.place(pe, on=spmw.mesh(MacIO, (M, N)))
        spmw.stream_in(A, into=P.west, index=(P.rows, ...))
        spmw.stream_in(B, into=P.north, index=(..., P.cols))
        spmw.gather(O, from_=P.c)

    with pytest.raises(spmw.SPMWBindingError, match="positional identity|uncovered"):
        spmw.elaborate(bad)


def test_unimplemented_knobs_are_refused_rather_than_ignored():
    """A knob this path does not honour must not be quietly dropped."""

    @spmw.fabric
    def folded(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        P = spmw.place(pe, on=spmw.mesh(MacIO, (M, N)), fold={1: 2})
        spmw.stream_in(A, into=P.west, index=(P.rows, ...))
        spmw.stream_in(B, into=P.north, index=(..., P.cols))
        spmw.gather(C, from_=P.c)

    with pytest.raises(spmw.SPMWPlacementError, match="does not realise"):
        spmw.build(folded, target="ref")

    @spmw.fabric
    def packed(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        P = spmw.place(pe, on=spmw.mesh(MacIO, (M, N)))
        spmw.stream_in(A, into=P.west, index=(P.rows, ...))
        spmw.stream_in(B, into=P.north, index=(..., P.cols))
        spmw.gather(C, from_=P.c, pack=lambda x: x)

    with pytest.raises(spmw.SPMWBindingError, match="pack= is not implemented"):
        spmw.elaborate(packed)


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------

KT, NT, MT = 4, 4, 6


class WsIO(spmw.Interface):
    a_in = spmw.In(int8)
    a_out = spmw.Out(int8)
    p_in = spmw.In(int32)
    p_out = spmw.Out(int32)
    w = spmw.MemIn(int8)


class ActIO(spmw.Interface):
    z_in = spmw.In(int32)
    y_out = spmw.Out(int8)


@spmw.unit
def mac(io: WsIO):
    for m in range(MT):
        a = io.a_in.get()
        p = io.p_in.get()
        io.p_out.put(p + a * io.w)
        io.a_out.put(a)


@spmw.unit
def act(io: ActIO):
    for m in range(MT):
        io.y_out.put(io.z_in.get())


def mxu_links(i, j):
    return {
        WsIO.a_out: spmw.to((i, j + 1), WsIO.a_in),
        WsIO.p_out: spmw.to((i + 1, j), WsIO.p_in),
    }


def _tpu_body(A, W, Y, two_writers=False):
    P = spmw.place(mac, on=spmw.Topology(WsIO, (KT, NT), link=mxu_links))
    Pact = spmw.place(act, on=spmw.Grid((NT,)))
    w_tile = spmw.mem(int8[KT, NT], layout=spmw.banked(on="col"), double=True)
    spmw.shard(w_tile, into=P.w)
    with spmw.phase("load"):
        spmw.copy(W, into=w_tile, how="dma")
        if two_writers:
            spmw.copy(W, into=w_tile, how="dma")
    with spmw.phase("compute"):
        spmw.stream_in(A, into=P.a_in, index=(..., P.rows))
        spmw.stream_in(0, into=P.p_in)
        spmw.link(P.p_out, to=Pact.z_in)
        (lane,) = Pact.axes
        spmw.gather(Y, from_=Pact.y_out, index=(..., lane))


def test_phases_order_bindings():
    @spmw.fabric
    def tpu(A: int8[MT, KT], W: int8[KT, NT], Y: int8[MT, NT]):
        _tpu_body(A, W, Y)

    graph = spmw.elaborate(tpu)
    assert graph.phases == ["load", "compute"]
    assert graph.bindings[0].phase is None  # the shard is a view, not an epoch
    assert graph.bindings[1].phase == "load"
    assert graph.bindings[-1].phase == "compute"


def test_two_writers_in_one_phase_is_rejected():
    @spmw.fabric
    def tpu(A: int8[MT, KT], W: int8[KT, NT], Y: int8[MT, NT]):
        _tpu_body(A, W, Y, two_writers=True)

    with pytest.raises(spmw.SPMWMemoryError, match="writers in phase"):
        spmw.elaborate(tpu)


# --------------------------------------------------------------------------
# Reconfiguration is an argument, not a rewrite
# --------------------------------------------------------------------------

R, C = 8, 8


def grouped_mxu(iface, shape, groups):
    rows, cols = shape
    d = cols // groups

    def link(i, j):
        links = {}
        if (j + 1) % d != 0:
            links[iface.a_out] = spmw.to((i, j + 1), iface.a_in)
        if i + 1 < rows:
            links[iface.p_out] = spmw.to((i + 1, j), iface.p_in)
        elif j + d < cols:
            links[iface.p_out] = spmw.to((0, j + d), iface.p_in)
        return links

    return spmw.Topology(iface, shape, link=link, name=f"grouped_mxu(G={groups})")


def attention_pv(groups):
    """The same five binding lines are correct for every grouping."""
    d = C // groups
    L = groups * R

    @spmw.fabric
    def pv(Pr: int8[MT, L], V: int8[L, d], Y: int8[MT, d]):
        P = spmw.place(mac, on=grouped_mxu(WsIO, (R, C), groups))
        Pa = spmw.place(act, on=spmw.Grid((d,)))
        k = P.rows
        g, e = spmw.split(P.cols, factor=groups)
        spmw.shard(V, into=P.w, index=(g * R + k, e))
        spmw.stream_in(Pr, into=P.a_in, index=(..., g * R + k))
        spmw.stream_in(0, into=P.p_in)
        spmw.link(P.p_out, to=Pa.z_in)
        (lane,) = Pa.axes
        spmw.gather(Y, from_=Pa.y_out, index=(..., lane))

    return pv


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_attention_pv_elaborates_for_every_grouping(groups):
    graph = spmw.elaborate(attention_pv(groups))
    assert len(graph.bindings) == 5
    P = graph.placements[0]
    assert len(P.a_in) == R * groups  # all columns fed, whatever the grouping
    assert len(P.p_in) == C // groups


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
