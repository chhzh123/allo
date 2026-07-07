# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo.spmw as spmw

M, N, K = 2, 2, 2


def _systolic_twin():
    """The SPMW twin of the flagship systolic GEMM: one interior unit + auto-halo."""
    grid = spmw.mesh((M, N))

    @spmw.unit
    def pe(ctx):
        c = 0
        for _ in range(K):
            a = ctx.west.get()
            b = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        # c_local is a local result buffer (bound by stream_out), written by index, not a port
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A, B, C):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    return gemm, pe, grid


def test_systolic_twin_validates():
    gemm, _, grid = _systolic_twin()
    collection = spmw.validate(gemm)
    assert len(collection.maps) == 1
    assert collection.maps[0].topology is grid
    # two stream_in + one stream_out
    assert len(collection.streams) == 3


def test_build_validates_then_defers_codegen():
    gemm, _, _ = _systolic_twin()
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        spmw.build(gemm, target="simulator")


def test_roles_reference_declared_ports():
    grid = spmw.mesh((4, 4))

    @spmw.unit
    def pe(ctx):
        pass

    @pe.role("west")
    def pe_load(ctx):
        pass

    @pe.role("east", "south")
    def pe_drain(ctx):
        pass

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid)

    spmw.validate(r)
    assert [edges for edges, _ in pe.roles] == [("west",), ("east", "south")]


def test_role_on_undeclared_port_rejected():
    grid = spmw.mesh((4, 4))

    @spmw.unit
    def pe(ctx):
        pass

    @pe.role("upward")  # not a mesh port
    def pe_x(ctx):
        pass

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid)

    with pytest.raises(spmw.SPMWError, match="undeclared port"):
        spmw.validate(r)


def test_stream_flow_requires_matching_ports():
    r_topo = spmw.ring(4)  # ports next/prev, no east/west

    @spmw.unit
    def pe(ctx):
        pass

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=r_topo)
        spmw.stream_in(A, into=pe, flow="W->E")  # needs east/west

    with pytest.raises(spmw.SPMWError, match="needs port"):
        spmw.validate(r)


def test_build_rejects_asymmetric_topology_at_build():
    def link(i, j):
        return {"east": ((i, j + 1), "west")}  # asymmetric

    topo = spmw.Topology(grid=(3, 3), link=link)

    @spmw.unit
    def pe(ctx):
        pass

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=topo)

    # rejected before codegen is ever reached
    with pytest.raises(spmw.SPMWError, match="asymmetric"):
        spmw.build(r, target="simulator")


def test_depths_override_recorded():
    grid = spmw.mesh((3, 3))

    @spmw.unit
    def pe(ctx):
        pass

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid, depths={"east": 4})

    collection = spmw.validate(r)
    assert collection.maps[0].depths == {"east": 4}


def test_depths_on_undeclared_port_rejected():
    grid = spmw.mesh((3, 3))

    @spmw.unit
    def pe(ctx):
        pass

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid, depths={"nope": 4})

    with pytest.raises(spmw.SPMWError, match="depths references undeclared"):
        spmw.validate(r)


def test_map_requires_a_unit():
    @spmw.region()
    def r(A):
        spmw.map(object(), grid=spmw.mesh((2, 2)))

    with pytest.raises(spmw.SPMWError, match="expects an @spmw.unit"):
        spmw.validate(r)


def test_region_without_map_rejected():
    @spmw.region()
    def r(A):
        pass

    with pytest.raises(spmw.SPMWError, match="no spmw.map"):
        spmw.validate(r)


def test_undeclared_port_io_in_body_rejected():
    grid = spmw.mesh((2, 2))

    @spmw.unit
    def pe(ctx):
        ctx.not_a_port.get()  # not a declared mesh port

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid)

    with pytest.raises(spmw.SPMWError, match="undeclared port"):
        spmw.validate(r)


def test_undeclared_port_io_in_role_body_rejected():
    grid = spmw.mesh((4, 4))

    @spmw.unit
    def pe(ctx):
        pass

    @pe.role("west")
    def pe_load(ctx):
        ctx.eastward.put(1)  # not a declared mesh port

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid)

    with pytest.raises(spmw.SPMWError, match="undeclared port"):
        spmw.validate(r)


def test_literal_ctx_port_in_body_checked():
    grid = spmw.mesh((2, 2))

    @spmw.unit
    def pe(ctx):
        ctx.port("bogus").get()

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid)

    with pytest.raises(spmw.SPMWError, match="undeclared port"):
        spmw.validate(r)


def test_dynamic_ctx_port_in_body_rejected():
    grid = spmw.mesh((2, 2))

    @spmw.unit
    def pe(ctx):
        name = "east"
        ctx.port(name).put(1)  # non-literal port name cannot be statically checked

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid)

    with pytest.raises(spmw.SPMWError, match="string-literal"):
        spmw.validate(r)


def test_stream_target_none_rejected():
    grid = spmw.mesh((2, 2))

    @spmw.unit
    def pe(ctx):
        ctx.west.get()

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=None, flow="W->E")

    with pytest.raises(spmw.SPMWError, match="needs a target"):
        spmw.validate(r)


def test_stream_into_unmapped_unit_rejected():
    grid = spmw.mesh((2, 2))

    @spmw.unit
    def pe(ctx):
        ctx.west.get()

    @spmw.unit
    def other(ctx):
        pass

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=other, flow="W->E")  # 'other' is never mapped

    with pytest.raises(spmw.SPMWError, match="not mapped"):
        spmw.validate(r)


def test_unknown_flow_rejected():
    grid = spmw.mesh((2, 2))

    @spmw.unit
    def pe(ctx):
        ctx.west.get()

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W=>E")  # typo, not a known flow

    with pytest.raises(spmw.SPMWError, match="unknown stream flow"):
        spmw.validate(r)


def test_default_port_depth_is_two_with_override():
    grid = spmw.mesh((3, 3))

    @spmw.unit
    def pe(ctx):
        ctx.west.get()

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid, depths={"east": 4})

    collection = spmw.validate(r)
    assert collection.maps[0].port_depths == {
        "east": 4,
        "west": 2,
        "north": 2,
        "south": 2,
    }
