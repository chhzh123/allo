# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import pytest
import allo.spmw as spmw
from allo.ir.types import float32


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
    """``spmw.tree(n)`` is a binary reduction tree over ``n`` leaves: ``2n-1`` heap-ordered nodes. Every
    node declares ``up``/``left``/``right``; the root's ``up`` and each leaf's ``left``/``right`` point
    out of bounds, so ``role_partition`` sees **three** link-presence classes -- root ``{up}`` / internal
    ``{}`` / leaf ``{left, right}`` -- the correct role structure for a reduction tree (not one class).
    """
    topo = spmw.tree(4)
    assert topo.grid == (7,)  # 2*4 - 1
    topo.validate()  # in-bounds parent/child links are reciprocal; out-of-bounds ones are boundary
    assert topo.boundary_ports_at((0,)) == {"up"}  # root: result leaves the tree
    assert topo.boundary_ports_at((1,)) == set()  # internal
    assert topo.boundary_ports_at((3,)) == {
        "left",
        "right",
    }  # leaf: inputs enter the tree
    partition = spmw.role_partition(topo)
    assert len(partition) == 3 and spmw.role_count(topo) == 3
    assert sorted(len(v) for v in partition.values()) == [
        1,
        2,
        4,
    ]  # 1 root, 2 internal, 4 leaves
    assert topo.links_at((1,))["up"] == ((0,), "left")  # reciprocal in-bounds peer link
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


# --------------------------------------------------------------------------------------------------
# Emitted-IR proof: the key-form permutation generators lower to a real `spmw.map` and survive the
# `spmw-role-partition` pass; the scatter/gather crossbar and the non-affine heap tree fail closed.
# --------------------------------------------------------------------------------------------------


def _link_classes(module_text):
    """The `spmw.link_classes = array<i64: ...>` sizes the role-partition pass emits (per class)."""
    match = re.search(r"spmw\.link_classes = array<i64: ([^>]*)>", module_text)
    return [int(x) for x in match.group(1).split(",")] if match else None


def test_butterfly_lowers_to_spmw_map():
    """A region mapping a passthrough over `spmw.butterfly(8)` lowers to ONE `spmw.map` with the (3,4)
    grid and a `#spmw.key_link` over the `lane` family, and `spmw-role-partition` emits a single
    link-presence class of all 12 butterflies (the O(#roles) body) -- the emitted-IR proof, not just a
    Python-object check."""

    @spmw.unit
    def relay(ctx):
        a: float32 = ctx.a.get()
        b: float32 = ctx.b.get()
        ctx.y.put(a)
        ctx.z.put(b)

    @spmw.region()
    def net(X: float32[8], Y: float32[8]):
        spmw.map(relay, grid=(3, 4), topo=spmw.butterfly(8))
        spmw.stream_in(X, into="lane", at_stage=0)
        spmw.stream_out(Y, from_="lane", at_stage=3)

    module = spmw.lower(net)
    text = str(module)
    assert text.count("spmw.map") == 1 and "grid = [3, 4]" in text
    assert (
        'key = "lane"' in text
    )  # key-form lane interconnect survives to the rolled op
    spmw._run_module_pass(module, "spmw-role-partition")
    assert _link_classes(str(module)) == [12]  # one role body over all 3*4 butterflies


def test_bitonic_lowers_to_spmw_map():
    """A region over `spmw.bitonic(8)` lowers to one `spmw.map` with the (6,4) grid + `lane` key link;
    `spmw-role-partition` emits a single class of all 24 comparators."""

    @spmw.unit
    def compare(ctx):
        a: float32 = ctx.a.get()
        b: float32 = ctx.b.get()
        ctx.hi.put(a)
        ctx.lo.put(b)

    @spmw.region()
    def net(X: float32[8], Y: float32[8]):
        spmw.map(compare, grid=(6, 4), topo=spmw.bitonic(8))
        spmw.stream_in(X, into="lane", at_stage=0)
        spmw.stream_out(Y, from_="lane", at_stage=6)

    module = spmw.lower(net)
    text = str(module)
    assert text.count("spmw.map") == 1 and "grid = [6, 4]" in text
    assert 'key = "lane"' in text
    spmw._run_module_pass(module, "spmw-role-partition")
    assert _link_classes(str(module)) == [24]  # one role body over all 6*4 comparators


def test_crossbar_lowering_fails_closed_without_boundary_keys():
    """The crossbar's per-input/-output scatter/gather boundary keys (`("in", i)` / `("out", j)`) cannot
    be supplied by the lane-family stream API, so lowering a crossbar region **fails closed** (a
    half-open scatter) rather than emitting an unfed interconnect."""

    @spmw.unit
    def switch(ctx):
        v: float32 = ctx.row_in.get()
        ctx.col_out.put(v)

    @spmw.region()
    def net(X: float32[4], Y: float32[4]):
        spmw.map(switch, grid=(4, 4), topo=spmw.crossbar(4))
        spmw.stream_in(X, into="in")
        spmw.stream_out(Y, from_="out")

    with pytest.raises(spmw.SPMWError):
        spmw.lower(net)


def test_tree_lowering_fails_closed_non_affine():
    """The heap tree's parent map `(i-1)//2` is not a constant translation (each node has a different
    parent offset), so it is not lowerable to the affine-`peer_link` rolled IR: `spmw.lower` **fails
    closed** with a clear diagnostic. The frontend role structure is still correct (3 classes above); a
    per-coordinate peer-edge IR is future work."""

    @spmw.unit
    def reduce_node(ctx):
        left_val: float32 = ctx.left.get()
        right_val: float32 = ctx.right.get()
        ctx.up.put(left_val + right_val)

    @spmw.region()
    def net(X: float32[4], Y: float32[1]):
        spmw.map(reduce_node, grid=(7,), topo=spmw.tree(4))

    with pytest.raises(spmw.SPMWError, match="affine-translation"):
        spmw.lower(net)
