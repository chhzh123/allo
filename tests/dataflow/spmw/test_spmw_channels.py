# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo.spmw as spmw


def test_mesh_resolves_to_two_families():
    families = spmw.resolve_channels(spmw.mesh((4, 4)))
    assert set(families) == {"east/west", "north/south"}
    # a mesh (M, N) has M*(N-1) horizontal and (M-1)*N vertical channels
    assert len(families["east/west"]) == 12
    assert len(families["north/south"]) == 12


def test_family_count_constant_but_instances_scale():
    small = spmw.resolve_channels(spmw.mesh((4, 4)))
    large = spmw.resolve_channels(spmw.mesh((8, 8)))
    # the number of FIFO families is constant as the grid scales...
    assert len(small) == 2
    assert len(large) == 2
    # ...while the number of channel instances grows with the grid
    assert len(large["east/west"]) == 56  # 8 * 7
    assert len(large["north/south"]) == 56  # 7 * 8


def test_channel_endpoints_are_reciprocal_peers():
    topology = spmw.mesh((4, 4))
    for channels in spmw.resolve_channels(topology).values():
        for src_coord, src_port, sink_coord, sink_port in channels:
            target = topology.links_at(src_coord)[src_port]
            assert tuple(target[0]) == sink_coord
            assert target[1] == sink_port


def test_one_dimensional_chain_has_one_family():
    families = spmw.resolve_channels(spmw.mesh((5,)))
    assert set(families) == {"next/prev"}
    assert len(families["next/prev"]) == 4  # 4 edges between 5 nodes


def test_grid_without_links_has_no_channels():
    assert spmw.resolve_channels(spmw.Grid((3, 3))) == {}


def test_key_form_channel_groups_by_key():
    def link(i):
        return {"out": (("c", 0), "src")} if i == 0 else {"in": (("c", 0), "sink")}

    families = spmw.resolve_channels(spmw.Topology(grid=(2,), link=link))
    assert len(families) == 1
    (channels,) = families.values()
    assert len(channels) == 1  # one src + one sink -> one channel


def test_resolve_channels_rejects_non_topology():
    with pytest.raises(spmw.SPMWError, match="expects an spmw.Topology"):
        spmw.resolve_channels(object())
