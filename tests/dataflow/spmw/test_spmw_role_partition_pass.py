# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo.spmw as spmw


def _mesh_region_with_roles(shape):
    grid = spmw.mesh(shape)

    @spmw.unit
    def pe(ctx):
        a = ctx.west.get()
        ctx.east.put(a)

    @pe.role("west")
    def pe_load(ctx):
        pass

    @pe.role("east", "south")
    def pe_drain(ctx):
        pass

    @spmw.region()
    def gemm(A):
        spmw.map(pe, grid=grid)

    return gemm


def _partitioned(shape):
    module = spmw.lower(_mesh_region_with_roles(shape))
    spmw._run_module_pass(module, "spmw-role-partition")
    return str(module)


def test_role_partition_pass_counts_grid_points():
    # roles are [interior, west, (east, south)]; on a 4x4 mesh the west column (4) is the load role,
    # the SE corner (1) is the drain role, the remaining 11 are interior
    text = _partitioned((4, 4))
    assert "spmw.partition = array<i64: 11, 4, 1>" in text


def test_role_partition_body_count_constant_across_grid():
    # the number of role bodies (partition entries) stays O(#roles) = 3 as the grid scales; only the
    # per-role instance counts grow -- the synthesis-time-win representation
    small = _partitioned((4, 4))
    large = _partitioned((8, 8))
    assert "spmw.partition = array<i64: 11, 4, 1>" in small
    # 8x8 = 64: west column 8, SE corner 1, interior 55
    assert "spmw.partition = array<i64: 55, 8, 1>" in large


def test_role_partition_interior_only_is_full_grid():
    # the systolic-style single-interior-role map assigns every grid point to the one role
    grid = spmw.mesh((3, 3))

    @spmw.unit
    def pe(ctx):
        pass

    @spmw.region()
    def top(A):
        spmw.map(pe, grid=grid)

    module = spmw.lower(top)
    spmw._run_module_pass(module, "spmw-role-partition")
    assert "spmw.partition = array<i64: 9>" in str(module)
