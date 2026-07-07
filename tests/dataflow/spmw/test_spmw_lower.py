# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo.spmw as spmw


def _mesh_region_with_roles(shape):
    grid = spmw.mesh(shape)

    @spmw.unit
    def pe(ctx):
        a = ctx.west.get()
        ctx.east.put(a)

    @pe.role("west")
    def pe_load(ctx):
        ctx.east.put(1)

    @pe.role("east", "south")
    def pe_drain(ctx):
        ctx.west.get()

    @spmw.region()
    def gemm(A):
        spmw.map(pe, grid=grid)

    return gemm


def test_lower_one_map_and_role_funcs():
    text = str(spmw.lower(_mesh_region_with_roles((4, 4))))
    # exactly one rolled map op
    assert text.count("spmw.map") == 1
    # one role func per predicate tag: interior + west + (east, south) = 3
    assert text.count("func.func @gemm_pe_") == 3
    # the typed topology attribute prints positionally on the op
    assert "topology = <" in text
    assert "grid = [4, 4]" in text


def test_role_count_independent_of_grid_size():
    small = str(spmw.lower(_mesh_region_with_roles((4, 4))))
    large = str(spmw.lower(_mesh_region_with_roles((8, 8))))
    # the role-func count and map count do NOT grow with the grid (no P0*P1 blow-up)
    assert small.count("func.func @gemm_pe_") == 3
    assert large.count("func.func @gemm_pe_") == 3
    assert small.count("spmw.map") == 1
    assert large.count("spmw.map") == 1
    assert "grid = [4, 4]" in small
    assert "grid = [8, 8]" in large


def test_topology_carries_four_peer_links():
    text = str(spmw.lower(_mesh_region_with_roles((4, 4))))
    assert text.count("spmw.peer_link") == 4


def test_build_ir_target_returns_module():
    module = spmw.build(_mesh_region_with_roles((4, 4)), target="ir")
    assert "spmw.map" in str(module)


def test_build_execution_target_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        spmw.build(_mesh_region_with_roles((4, 4)), target="simulator")


def test_grid_only_region_lowers_with_empty_links():
    @spmw.unit
    def pe(ctx):
        pass

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=spmw.Grid((3, 3)))

    text = str(spmw.lower(r))
    assert "links = []" in text
    assert text.count("func.func @r_pe_") == 1  # interior role only
