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


def _resolved(shape):
    module = spmw.lower(_mesh_region_with_roles(shape))
    spmw._run_module_pass(module, "spmw-resolve-channels")
    return str(module)


def test_resolve_channels_pass_groups_peer_families():
    # a 2-D mesh's peer links resolve into two undirected channel families: east/west and
    # north/south (a link and its reciprocal collapse into one family), each with its FIFO depth
    text = _resolved((4, 4))
    assert 'spmw.channel_families = ["east/west", "north/south"]' in text
    # the per-family FIFO depths the emitter declares (the peer-link depth, default 2)
    assert "spmw.channel_family_depths = array<i64: 2, 2>" in text


def test_resolve_channels_family_count_constant_across_grid():
    # the family count (the FIFO arrays HLS declares) is constant as the grid scales, even though the
    # channel instance count is O(P0*P1)
    assert 'spmw.channel_families = ["east/west", "north/south"]' in _resolved((4, 4))
    assert 'spmw.channel_families = ["east/west", "north/south"]' in _resolved((8, 8))


def _interior_mesh(shape):
    grid = spmw.mesh(shape)

    @spmw.unit
    def pe(ctx):
        pass

    @spmw.region()
    def top(A):
        spmw.map(pe, grid=grid)

    return top


def _link_classes(shape):
    module = spmw.lower(_interior_mesh(shape))
    spmw._run_module_pass(module, "spmw-role-partition")
    text = str(module)
    import re

    return [
        int(x)
        for x in re.search(r"spmw\.link_classes = array<i64: ([^>]*)>", text)
        .group(1)
        .split(",")
    ]


def test_link_presence_classes_are_nine_for_any_mesh():
    # the AC-4 link-presence classification, computed by the named pass independent of declared roles:
    # a 2-D mesh (extents >= 3) has exactly nine classes (interior, 4 edges, 4 corners), constant as
    # the grid scales; only the per-class instance counts grow
    small = _link_classes((4, 4))
    large = _link_classes((8, 8))
    assert len(small) == 9 and len(large) == 9
    assert sum(small) == 16 and sum(large) == 64
    # interior class (empty signature, sorts first) scales with the interior area (2x2 vs 6x6)
    assert small[0] == 4 and large[0] == 36


def test_link_presence_degenerate_grid_has_fewer_classes():
    # a thin grid has fewer link-presence classes than the full nine
    assert len(_link_classes((2, 4))) == 6
    assert len(_link_classes((2, 2))) == 4


def _checkerboard_twin(n=4):
    from allo.ir.types import float32

    grid = spmw.mesh((n, n))

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

    @pe.variant("(d0 + d1) mod 2")
    def pe_odd(ctx):
        c: float32 = 0
        for k in range(4):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a + b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: float32[n, 4], B: float32[4, n], C: float32[n, n]):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    return gemm


def test_predicate_variants_split_partition_and_carry_class_keys():
    import re

    module = spmw.lower(_checkerboard_twin(4))
    spmw._run_module_pass(module, "spmw-role-partition")
    ir = str(module)
    # the base interior and the odd variant are DISTINCT compute roles: the partition has two entries
    # that split the 4x4 grid evenly along the checkerboard (they are not merged into one role)
    counts = [
        int(x)
        for x in re.search(r"spmw\.partition = array<i64: ([^>]*)>", ir)
        .group(1)
        .split(",")
    ]
    assert len(counts) == 2 and sorted(counts) == [8, 8]
    # the class-key identities are carried in the IR (not just counts): a predicated class records
    # (selected-role index, tag) as a #<role>:<tag> suffix, so distinct predicate roles stay distinct
    # classes and never collapse. Here the base interior class is "" and the variant (role 1, tag 1)
    # class is "#1:1".
    keys = re.search(r"spmw\.link_class_keys = \[([^\]]*)\]", ir).group(1)
    assert '"#1:1"' in keys and '""' in keys
