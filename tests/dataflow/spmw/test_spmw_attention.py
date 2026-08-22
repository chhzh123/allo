# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention's P.V GEMM: reconfigurable tensor parallelism, reduction as wiring.

A small head dimension leaves half a weight-stationary array idle.  Cutting the
array into column slabs and routing each slab's partial-sum chain into the top of
the next fills it -- and because the psum chain already *is* an adder chain, the
reduction needs no new component, only five lines in the link rule.

The PE and the activation unit are the mini-TPU's, unchanged; ``G`` is a Python
argument, not a structural rewrite.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int8, int32

R, C = 4, 4  # the physical array
MT = 6  # rows streamed through it
SHIFT = 2


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
        z = io.z_in.get()
        if z < 0:
            z = 0
        y: int8 = z >> SHIFT
        io.y_out.put(y)


def grouped_mxu(iface, shape, groups):
    """Column slabs; the psum chain serpentines into the next slab's top."""
    rows, cols = shape
    d = cols // groups

    def link(i, j):
        links = {}
        if (j + 1) % d != 0:  # forward activations inside a slab only
            links[iface.a_out] = spmw.to((i, j + 1), iface.a_in)
        if i + 1 < rows:  # psums: down my column, ...
            links[iface.p_out] = spmw.to((i + 1, j), iface.p_in)
        elif j + d < cols:  # ... then on to the next slab's top -- the reduction
            links[iface.p_out] = spmw.to((0, j + d), iface.p_in)
        return links

    return spmw.Topology(iface, shape, link=link, name=f"grouped_mxu(G={groups})")


def attention_pv(groups):
    """Configuration is a Python argument."""
    d = C // groups  # output columns per slab
    L = groups * R  # sequence tile covered fully spatially

    @spmw.fabric
    def pv(Pr: int8[MT, L], V: int8[L, d], Y: int8[MT, d]):
        P = spmw.place(mac, on=grouped_mxu(WsIO, (R, C), groups))
        Pa = spmw.place(act, on=spmw.Grid((d,)))
        k = P.rows  # row axis symbol, extent R
        g, e = spmw.split(P.cols, factor=groups)  # g: which slab; e: column in it
        spmw.shard(V, into=P.w, index=(g * R + k, e))  # PE (k,c) holds V[g.R+k, e]
        spmw.stream_in(Pr, into=P.a_in, index=(..., g * R + k))
        spmw.stream_in(0, into=P.p_in)  # unbound p_in = slab 0's top row only
        spmw.link(P.p_out, to=Pa.z_in)  # unbound p_out = last slab's bottom row
        (lane,) = Pa.axes
        spmw.gather(Y, from_=Pa.y_out, index=(..., lane))

    return pv


def _reference(Pr, V):
    z = Pr.astype(np.int32) @ V.astype(np.int32)
    return (np.maximum(z, 0) >> SHIFT).astype(np.int8)


def _operands(groups, seed=7):
    rng = np.random.default_rng(seed)
    d, L = C // groups, groups * R
    Pr = rng.integers(-4, 4, (MT, L)).astype(np.int8)
    V = rng.integers(-4, 4, (L, d)).astype(np.int8)
    return Pr, V


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_boundaries_are_computed_not_declared(groups):
    """Withholding edges reshapes the open sets; the bindings never change."""
    graph = spmw.elaborate(attention_pv(groups))
    P = graph.placements[0]
    d = C // groups
    assert len(P.a_in) == R * groups  # one slab-west column per slab
    assert len(P.p_in) == d  # seeded at slab 0's top only
    assert len(P.p_out) == d  # drained at the last slab's bottom only
    assert len(graph.bindings) == 5  # the same five lines for every grouping


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_reference_matches(groups):
    Pr, V = _operands(groups)
    Y = np.zeros((MT, C // groups), dtype=np.int8)
    spmw.build(attention_pv(groups), target="ref")(Pr, V, Y)
    np.testing.assert_array_equal(Y, _reference(Pr, V))


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_simulator_matches(groups):
    Pr, V = _operands(groups)
    Y = np.zeros((MT, C // groups), dtype=np.int8)
    spmw.build(attention_pv(groups), target="simulator")(Pr, V, Y)
    np.testing.assert_array_equal(Y, _reference(Pr, V))


def test_the_degenerate_case_costs_nothing():
    """G=1 has no slab edges: the chain simply ends at the one bottom row."""
    one = spmw.elaborate(attention_pv(1)).placements[0]
    assert len(one.a_in) == R  # a single west column
    assert len(one.p_out) == C  # every column drains


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
