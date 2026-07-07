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


def test_key_channel_two_src_rejected():
    def link(i):
        return {"a": (("k",), "src"), "b": (("k",), "src")}

    with pytest.raises(spmw.SPMWError, match="exactly one source"):
        spmw.Topology(grid=(1,), link=link).validate()


def test_key_channel_zero_sink_rejected():
    def link(i):
        return {"a": (("k",), "src")}

    with pytest.raises(spmw.SPMWError, match="sink"):
        spmw.Topology(grid=(1,), link=link).validate()


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
