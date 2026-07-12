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


# --------------------------------------------------------------------------------------------------
# M6 non-mesh topology generators: butterfly / bitonic / crossbar / tree. Each builds on the same
# key-/peer-link machinery as mesh/ring/scatter/gather; the tests build, run the static checks, assert
# structure, tie the network to the emitted spmw.map role count (O(#roles)), and fail closed on bad input.
# --------------------------------------------------------------------------------------------------


def _upper(stage, butterfly):
    """The radix-2 butterfly's upper lane index (== the FFT twin's ``get_upper_idx``)."""
    span = 1 << stage
    return (butterfly // span) * (span << 1) + (butterfly % span)


def test_butterfly_matches_fft_interconnect():
    """``spmw.butterfly(n)`` is the ``(log2 n, n/2)`` radix-2 butterfly network the FFT twin wires by
    hand: it validates, every internal lane channel is a 1->1 peer, each butterfly touches the expected
    upper/lower lanes, and the emitted ``spmw.map`` is one role body (O(#roles))."""
    topo = spmw.butterfly(8)
    assert topo.grid == (3, 4)  # (log2 8, 8/2)
    srcs, sinks = spmw.boundary_lane_keys(8)
    topo.validate(external_srcs=srcs, external_sinks=sinks)
    roles = topo.key_channel_roles(srcs, sinks)
    assert roles and all(
        r == "peer" for r in roles.values()
    )  # a permutation network is all 1->1
    for stage in range(3):
        for but in range(4):
            links = topo.links_at((stage, but))
            up = _upper(stage, but)
            assert links["a"] == (("lane", stage, up), "sink")
            assert links["b"] == (("lane", stage, up + (1 << stage)), "sink")
            assert links["y"] == (("lane", stage + 1, up), "src")
            assert links["z"] == (("lane", stage + 1, up + (1 << stage)), "src")
    assert (
        spmw.role_count(topo) == 1
    )  # one butterfly body -> the O(#roles) rolled spmw.map


def test_bitonic_sorting_network_structure():
    """``spmw.bitonic(n)`` is the bitonic sorter: ``log2(n)*(log2(n)+1)/2`` comparator stages of ``n/2``
    compare-exchange units; each stage is a perfect matching of the ``n`` lanes; validates as an
    all-peer network with a single role body."""
    topo = spmw.bitonic(8)
    n_stages = 3 * (3 + 1) // 2  # log2(8)*(log2(8)+1)/2 = 6
    assert topo.grid == (n_stages, 4)
    srcs, sinks = spmw.boundary_lane_keys(8, stages=n_stages)
    topo.validate(external_srcs=srcs, external_sinks=sinks)
    roles = topo.key_channel_roles(srcs, sinks)
    assert all(r == "peer" for r in roles.values())
    for stage in range(n_stages):  # every lane compared exactly once per stage
        lanes = []
        for comp in range(4):
            links = topo.links_at((stage, comp))
            lanes += [links["a"][0][2], links["b"][0][2]]
        assert sorted(lanes) == list(range(8))
    assert spmw.role_count(topo) == 1


def test_crossbar_scatter_gather():
    """``spmw.crossbar(n)`` is an ``n x n`` switch matrix: each input row scatters over its switches and
    each output column gathers them, so it classifies as ``n`` scatters + ``n`` gathers over one switch
    body."""
    topo = spmw.crossbar(4)
    assert topo.grid == (4, 4)
    srcs, sinks = spmw.crossbar_boundary_keys(4)
    topo.validate(external_srcs=srcs, external_sinks=sinks)
    roles = topo.key_channel_roles(srcs, sinks)
    ins = {k: r for k, r in roles.items() if k[0] == "in"}
    outs = {k: r for k, r in roles.items() if k[0] == "out"}
    assert len(ins) == 4 and all(r == "scatter" for r in ins.values())
    assert len(outs) == 4 and all(r == "gather" for r in outs.values())
    assert spmw.role_count(topo) == 1
    with pytest.raises(
        spmw.SPMWError
    ):  # without the external input/output endpoints it is half-open
        topo.validate()


def test_tree_reduction_structure():
    """``spmw.tree(n)`` is a binary reduction tree over ``n`` leaves: ``2n-1`` heap-ordered nodes with
    reciprocal peer parent/child links; the root has no parent and leaves have no children; peer
    symmetry validates."""
    topo = spmw.tree(4)
    assert topo.grid == (7,)  # 2*4 - 1
    topo.validate()
    assert set(topo.links_at((0,))) == {"left", "right"}  # root: children only
    assert set(topo.links_at((1,))) == {
        "up",
        "left",
        "right",
    }  # internal: parent + children
    for leaf in (3, 4, 5, 6):
        assert set(topo.links_at((leaf,))) == {"up"}  # leaves: parent only
    assert topo.links_at((1,))["up"] == ((0,), "left")  # reciprocal peer link
    assert topo.links_at((0,))["left"] == ((1,), "up")


def test_topology_generators_reject_bad_input():
    """Hardening: the generators fail closed on malformed sizes, and ``Topology.validate`` rejects a
    malformed (asymmetric peer / many-to-many key) topology before any backend emission.
    """
    for gen in (spmw.butterfly, spmw.bitonic, spmw.tree):
        with pytest.raises(spmw.SPMWError):
            gen(6)  # not a power of two
        with pytest.raises(spmw.SPMWError):
            gen(1)  # too small
    with pytest.raises(spmw.SPMWError):
        spmw.crossbar(0)  # non-positive size
    # an asymmetric peer topology is rejected.
    bad_peer = spmw.Topology(grid=(2,), link=lambda i: {"x": ((i + 1,), "y")})
    with pytest.raises(spmw.SPMWError, match="asymmetric"):
        bad_peer.validate()
    # a many-to-many key channel (2 sources AND 2 sinks) is rejected.
    bad_key = spmw.Topology(
        grid=(4,), link=lambda i: {"p": ("bus", "src" if i < 2 else "sink")}
    )
    with pytest.raises(spmw.SPMWError):
        bad_key.validate()
