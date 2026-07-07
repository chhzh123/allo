# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo.spmw as spmw


def test_two_d_mesh_has_nine_roles_at_any_size():
    # {interior, 4 edges, 4 corners} = 9, constant as the grid scales
    for shape in [(3, 3), (4, 4), (8, 8), (16, 16), (32, 32)]:
        assert spmw.role_count(spmw.mesh(shape)) == 9


def test_degenerate_grids_have_smaller_role_sets():
    assert spmw.role_count(spmw.mesh((2, 4))) == 6  # two rows, both on a boundary
    assert spmw.role_count(spmw.mesh((2, 2))) == 4  # four corners only
    assert spmw.role_count(spmw.mesh((5,))) == 3  # 1-D: interior + two ends


def test_partition_covers_grid_disjointly():
    part = spmw.role_partition(spmw.mesh((6, 6)))
    coords = [c for points in part.values() for c in points]
    assert len(coords) == 36  # covers the whole grid
    assert len(set(coords)) == 36  # the groups are disjoint
    assert len(part[()]) == 16  # interior = (6-2) * (6-2)


def test_partition_signatures_are_missing_links():
    part = spmw.role_partition(spmw.mesh((4, 4)))
    assert () in part  # interior: nothing missing
    assert ("west",) in part  # west edge: missing its west neighbor
    assert ("north", "west") in part  # NW corner: missing north and west
    assert len(part) == 9


def test_role_count_rejects_non_topology():
    with pytest.raises(spmw.SPMWError, match="expects an spmw.Topology"):
        spmw.role_count("not a topology")
