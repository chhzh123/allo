# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo.spmw as spmw


def _region(shape):
    grid = spmw.mesh(shape)

    @spmw.unit
    def pe(ctx):
        ctx.west.get()

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


def test_unroll_one_call_per_grid_point():
    text = str(spmw.unroll(_region((4, 4))))
    assert text.count("call @gemm_pe_") == 16  # P0*P1 = 16 per-PID calls


def test_role_func_count_is_constant_across_grid_sizes():
    small = str(spmw.unroll(_region((4, 4))))
    large = str(spmw.unroll(_region((6, 6))))
    # the role func BODIES stay O(#roles) = 3, never cloned per grid point
    assert small.count("func.func @gemm_pe_") == 3
    assert large.count("func.func @gemm_pe_") == 3
    # ... while the per-PID call count scales with the grid
    assert small.count("call @gemm_pe_") == 16
    assert large.count("call @gemm_pe_") == 36


def test_role_assignment_by_missing_links():
    text = str(spmw.unroll(_region((4, 4))))
    # the west column (4 PIDs) is missing its west link -> the load role;
    # the south-east corner is missing east and south -> the drain role;
    # every other grid point -> interior.
    assert text.count("call @gemm_pe_west") == 4
    assert text.count("call @gemm_pe_east_south") == 1
    assert text.count("call @gemm_pe_interior") == 11


def test_build_unroll_target():
    text = str(spmw.build(_region((4, 4)), target="unroll"))
    assert text.count("call @gemm_pe_") == 16


def test_ambiguous_role_rejected():
    grid = spmw.mesh((4, 4))

    @spmw.unit
    def pe(ctx):
        pass

    @pe.role("west")
    def west_role(ctx):
        pass

    @pe.role("north")
    def north_role(ctx):
        pass

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid)

    # the NW corner is missing both west and north; two size-1 roles fit -> ambiguous
    with pytest.raises(spmw.SPMWError, match="ambiguous role"):
        spmw.unroll(r)
