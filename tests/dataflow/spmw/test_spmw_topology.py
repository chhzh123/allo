# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo.spmw as spmw


def test_mesh_2d_ports_and_validate():
    m = spmw.mesh((3, 3))
    assert m.dims == 2
    assert m.port_names() == {"east", "west", "north", "south"}
    m.validate()


def test_mesh_1d_chain_ports():
    m = spmw.mesh((5,))
    assert m.port_names() == {"next", "prev"}
    m.validate()


def test_ring_is_symmetric_with_no_boundary():
    r = spmw.ring(4)
    r.validate()
    # wrap-around means every node has both neighbors in range
    assert r.boundary_ports_at((0,)) == set()


def test_mesh_boundary_ports():
    m = spmw.mesh((3, 3))
    assert m.boundary_ports_at((0, 0)) == {"west", "north"}
    assert m.boundary_ports_at((1, 1)) == set()
    assert m.boundary_ports_at((2, 2)) == {"east", "south"}


def test_grid_has_no_links():
    g = spmw.Grid((4, 4))
    assert g.port_names() == set()
    g.validate()


def test_asymmetric_link_rejected():
    def link(i, j):
        # 'east' with no reciprocal 'west' on the peer
        return {"east": ((i, j + 1), "west")}

    topo = spmw.Topology(grid=(3, 3), link=link)
    with pytest.raises(spmw.SPMWError, match="asymmetric"):
        topo.validate()


def test_grid_rank_mismatch_rejected():
    # a two-argument link over a 1-D grid
    topo = spmw.Topology(grid=(4,), link=lambda i, j: {"east": ((i, j + 1), "west")})
    with pytest.raises(spmw.SPMWError):
        topo.validate()


def test_peer_coordinate_rank_mismatch_rejected():
    topo = spmw.Topology(grid=(4, 4), link=lambda i, j: {"x": ((i,), "y")})
    with pytest.raises(spmw.SPMWError, match="rank"):
        topo.validate()


def test_mesh_rejects_rank_three():
    with pytest.raises(spmw.SPMWError):
        spmw.mesh((2, 2, 2))


def test_key_channel_valid():
    def link(i):
        if i == 0:
            return {"out": (("c", 0), "src")}
        return {"in": (("c", 0), "sink")}

    spmw.Topology(grid=(2,), link=link).validate()


def test_key_channel_zero_sink_rejected():
    # a source with no sink is neither a scatter nor a gather -- rejected
    def link(i):
        return {"a": (("k",), "src")}

    with pytest.raises(spmw.SPMWError, match="0 sink"):
        spmw.Topology(grid=(1,), link=link).validate()


def test_key_channel_many_to_many_rejected():
    # two sources AND two sinks on one key is many-to-many -- not a supported collective
    def link(i):
        if i < 2:
            return {"o": (("k",), "src")}
        return {"i": (("k",), "sink")}

    with pytest.raises(spmw.SPMWError, match="source.*sink|sink.*source"):
        spmw.Topology(grid=(4,), link=link).validate()


def test_scatter_topology_classifies_as_scatter():
    # a scatter is one source fanning out to many sinks (a declared fan-out key)
    topo = spmw.scatter(4)
    assert topo.validate().key_channel_roles() == {"scatter": "scatter"}


def test_gather_topology_classifies_as_gather():
    # a gather is many sources fanning in to one sink (a declared fan-in key)
    topo = spmw.gather(4)
    assert topo.validate().key_channel_roles() == {"gather": "gather"}


def test_peer_key_channel_still_one_to_one():
    def link(i):
        return {"out": (("c",), "src")} if i == 0 else {"in": (("c",), "sink")}

    assert spmw.Topology(grid=(2,), link=link).validate().key_channel_roles() == {
        ("c",): "peer"
    }


def test_context_rank_and_ports():
    ctx = spmw.mesh((3, 3)).context((1, 1))
    assert ctx.rank() == (1, 1)
    assert ctx.port("east").name == "east"
    assert ctx.west.name == "west"


def test_context_rank_1d_is_scalar():
    ctx = spmw.mesh((5,)).context((2,))
    assert ctx.rank() == 2


def test_context_undeclared_port_rejected():
    ctx = spmw.mesh((3, 3)).context((1, 1))
    with pytest.raises(spmw.SPMWError, match="not declared"):
        ctx.port("nowhere")
    with pytest.raises(AttributeError):
        _ = ctx.nowhere


def test_shard_must_divide_grid():
    grid = spmw.mesh((4, 4))

    @spmw.unit
    def pe(ctx):
        pass

    with pytest.raises(spmw.SPMWError, match="divide"):
        spmw.map(pe, grid=grid, shard=(3, 2))


def test_shard_rank_must_match_grid():
    grid = spmw.mesh((4, 4))

    @spmw.unit
    def pe(ctx):
        pass

    with pytest.raises(spmw.SPMWError, match="rank"):
        spmw.map(pe, grid=grid, shard=(2,))


def test_shard_lowers_to_map_attr():
    from allo.ir.types import float32

    grid = spmw.mesh((4, 4))

    @spmw.unit
    def pe(ctx):
        c: float32 = 0
        for k in range(4):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: float32[4, 4], B: float32[4, 4], C: float32[4, 4]):
        spmw.map(pe, grid=grid, shard=(2, 2))
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    ir = str(spmw.lower(gemm))
    # the 2-level hierarchy (2x2 shards of the 4x4 grid) is recorded on the map op
    assert "spmw.shard = array<i64: 2, 2>" in ir
