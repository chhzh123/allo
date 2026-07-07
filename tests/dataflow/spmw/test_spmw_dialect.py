# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from allo._mlir.ir import Context, Location, Module
import allo._mlir.dialects.allo as allo_d


def _context():
    ctx = Context()
    # register_dialect loads both the allo and spmw dialects into the context
    allo_d.register_dialect(ctx)
    return ctx


def _parse(ir):
    ctx = _context()
    with ctx, Location.unknown():
        return Module.parse(ir)


# A valid 2x2 mesh: a peer link (east->west) plus one interior role.
VALID_PEER = """
module {
  func.func @pe_interior(%a: memref<2x2xf32>) {
    return
  }
  func.func @top(%A: memref<2x2xf32>, %B: memref<2x2xf32>) {
    spmw.map (%A, %B)
      topology = #spmw.topology<grid = [2, 2], dims = 2, links = [
        #spmw.peer_link<port = "east", map = affine_map<(i, j) -> (i, j + 1)>, peer = "west", depth = 2>
      ]>
      roles = [#spmw.role<unit = @pe_interior, missing = []>]
      : memref<2x2xf32>, memref<2x2xf32>
    return
  }
}
"""

# A valid 1D key channel: one src and one sink sharing key "c".
VALID_KEY = """
module {
  func.func @pe(%a: memref<2xf32>) {
    return
  }
  func.func @top(%A: memref<2xf32>) {
    spmw.map (%A)
      topology = #spmw.topology<grid = [2], dims = 1, links = [
        #spmw.key_link<port = "out", key = "c", end = "src", depth = 2>,
        #spmw.key_link<port = "in", key = "c", end = "sink", depth = 2>
      ]>
      roles = [#spmw.role<unit = @pe, missing = []>]
      : memref<2xf32>
    return
  }
}
"""


def test_spmw_rank_round_trips():
    module = _parse(
        """
        module {
          func.func @unit() {
            %0:2 = "spmw.rank"() : () -> (index, index)
            return
          }
        }
        """
    )
    assert "spmw.rank" in str(module)


def test_valid_peer_topology_round_trips():
    module = _parse(VALID_PEER)
    printed = str(module)
    assert "spmw.map" in printed and "peer_link" in printed
    # round-trip the pretty-printed form back through the parser
    assert "spmw.map" in str(_parse(printed))


def test_valid_key_topology_round_trips():
    module = _parse(VALID_KEY)
    assert "key_link" in str(module)


def test_grid_rank_mismatch_rejected():
    bad = VALID_PEER.replace("grid = [2, 2], dims = 2", "grid = [2, 2], dims = 3")
    with pytest.raises(Exception):
        _parse(bad)


def test_key_two_src_rejected():
    # both endpoints declare src for key "c": zero sink, two src
    bad = VALID_KEY.replace('end = "sink"', 'end = "src"')
    with pytest.raises(Exception):
        _parse(bad)


def test_bad_link_map_rank_rejected():
    # peer map has one dim/result but the topology is 2-D
    bad = VALID_PEER.replace(
        "map = affine_map<(i, j) -> (i, j + 1)>", "map = affine_map<(i) -> (i)>"
    )
    with pytest.raises(Exception):
        _parse(bad)


def test_non_positive_depth_rejected():
    bad = VALID_PEER.replace('peer = "west", depth = 2', 'peer = "west", depth = 0')
    with pytest.raises(Exception):
        _parse(bad)


def test_bad_role_payload_rejected():
    bad = VALID_PEER.replace(
        "roles = [#spmw.role<unit = @pe_interior, missing = []>]",
        'roles = ["not_a_role"]',
    )
    with pytest.raises(Exception):
        _parse(bad)


def test_missing_role_symbol_rejected():
    bad = VALID_PEER.replace("unit = @pe_interior", "unit = @does_not_exist")
    with pytest.raises(Exception):
        _parse(bad)


def test_bad_fold_rank_rejected():
    # fold has rank 1 but the topology is 2-D
    bad = VALID_PEER.replace(
        "roles = [#spmw.role<unit = @pe_interior, missing = []>]",
        "roles = [#spmw.role<unit = @pe_interior, missing = []>] {fold = array<i64: 4>}",
    )
    with pytest.raises(Exception):
        _parse(bad)
