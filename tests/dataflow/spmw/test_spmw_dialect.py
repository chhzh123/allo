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
        #spmw.peer_link<port = "east", map = affine_map<(i, j) -> (i, j + 1)>, peer = "west", depth = 2>,
        #spmw.peer_link<port = "west", map = affine_map<(i, j) -> (i, j - 1)>, peer = "east", depth = 2>
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


def test_role_abi_mismatch_rejected():
    # a role func's leading memref arg must match the map's tensor operand type
    bad = VALID_PEER.replace(
        "func.func @pe_interior(%a: memref<2x2xf32>)",
        "func.func @pe_interior(%a: memref<4x4xf32>)",
    )
    with pytest.raises(Exception):
        _parse(bad)


def test_role_with_memref_after_nonmemref_rejected():
    # a role must name all its map tensors (memref args) before any per-instantiation param;
    # a memref appearing after an index/stream is rejected
    bad = """
module {
  func.func @pe_interior(%p: index, %a: memref<2x2xf32>) {
    return
  }
  func.func @top(%A: memref<2x2xf32>) {
    spmw.map (%A)
      topology = #spmw.topology<grid = [2, 2], dims = 2, links = []>
      roles = [#spmw.role<unit = @pe_interior, missing = []>]
      : memref<2x2xf32>
    return
  }
}
"""
    with pytest.raises(Exception):
        _parse(bad)


def test_role_with_more_memrefs_than_tensors_rejected():
    # the role takes two memrefs but the map has only one tensor operand
    bad = """
module {
  func.func @pe_interior(%a: memref<2x2xf32>, %b: memref<2x2xf32>) {
    return
  }
  func.func @top(%A: memref<2x2xf32>) {
    spmw.map (%A)
      topology = #spmw.topology<grid = [2, 2], dims = 2, links = []>
      roles = [#spmw.role<unit = @pe_interior, missing = []>]
      : memref<2x2xf32>
    return
  }
}
"""
    with pytest.raises(Exception):
        _parse(bad)


def test_spmw_unroll_rejects_key_links():
    # key_link channels are resolved by rendezvous key, not by grid neighbor, so
    # the grid-expansion pass fails closed rather than silently dropping them.
    from allo._mlir.passmanager import PassManager

    module = _parse(VALID_KEY)
    with module.context:
        with pytest.raises(Exception, match="key_link"):
            PassManager.parse("builtin.module(spmw-unroll)").run(module.operation)


# A valid map with a west peer link, an interior role, and a west loader halo task.
VALID_HALO = """
module {
  func.func @pe(%a: memref<2x2xf32>, %pi: index, %pj: index, %w: !allo.stream<f32, 2>) {
    return
  }
  func.func @load_a(%a: memref<2x2xf32>, %p: index, %s: !allo.stream<f32, 2>) {
    return
  }
  func.func @top(%A: memref<2x2xf32>) {
    spmw.map (%A)
      topology = #spmw.topology<grid = [2, 2], dims = 2, links = [
        #spmw.peer_link<port = "west", map = affine_map<(i, j) -> (i, j - 1)>, peer = "east", depth = 2>,
        #spmw.peer_link<port = "east", map = affine_map<(i, j) -> (i, j + 1)>, peer = "west", depth = 2>
      ]>
      roles = [#spmw.role<unit = @pe, missing = [], ports = ["west"]>]
      {halo = [#spmw.halo<unit = @load_a, port = "west", kind = "load", axis = 0, operand = 0>]}
      : memref<2x2xf32>
    return
  }
}
"""


def test_valid_halo_round_trips():
    assert "spmw.halo" in str(_parse(VALID_HALO))


def test_halo_loader_wrong_abi_rejected():
    # a load halo must take (memref, index, stream); drop the index arg
    bad = VALID_HALO.replace(
        "func.func @load_a(%a: memref<2x2xf32>, %p: index, %s: !allo.stream<f32, 2>)",
        "func.func @load_a(%a: memref<2x2xf32>, %s: !allo.stream<f32, 2>)",
    )
    with pytest.raises(Exception, match="memref, index, stream"):
        _parse(bad)


def test_halo_port_not_peer_rejected():
    # "north" is not a declared peer-link port of the topology
    bad = VALID_HALO.replace(
        'port = "west", kind = "load"', 'port = "north", kind = "load"'
    )
    with pytest.raises(Exception, match="not a declared peer-link port"):
        _parse(bad)


def test_duplicate_halo_task_rejected():
    bad = VALID_HALO.replace(
        '{halo = [#spmw.halo<unit = @load_a, port = "west", kind = "load", axis = 0, operand = 0>]}',
        '{halo = [#spmw.halo<unit = @load_a, port = "west", kind = "load", axis = 0, operand = 0>, '
        '#spmw.halo<unit = @load_a, port = "west", kind = "load", axis = 0, operand = 0>]}',
    )
    with pytest.raises(Exception, match="duplicate halo task"):
        _parse(bad)


_ASYMMETRIC_PEER = """
module {
  func.func @pe(%a: memref<2x2xf32>) {
    return
  }
  func.func @top(%A: memref<2x2xf32>) {
    spmw.map (%A)
      topology = #spmw.topology<grid = [2, 2], dims = 2, links = [
        #spmw.peer_link<port = "east", map = affine_map<(i, j) -> (i, j + 1)>, peer = "west", depth = 2>
      ]>
      roles = [#spmw.role<unit = @pe, missing = []>]
      : memref<2x2xf32>
    return
  }
}
"""


def test_asymmetric_peer_link_rejected():
    # a one-way "east" peer with no reciprocal "west" link back
    with pytest.raises(Exception, match="no reciprocal"):
        _parse(_ASYMMETRIC_PEER)


def test_duplicate_peer_port_rejected():
    bad = VALID_PEER.replace(
        'port = "west", map = affine_map<(i, j) -> (i, j - 1)>, peer = "east"',
        'port = "east", map = affine_map<(i, j) -> (i, j - 1)>, peer = "east"',
    )
    with pytest.raises(Exception, match="duplicate peer link port"):
        _parse(bad)


def test_non_positive_grid_rejected():
    bad = VALID_PEER.replace("grid = [2, 2]", "grid = [0, 2]")
    with pytest.raises(Exception, match="grid extent must be positive"):
        _parse(bad)


def test_role_missing_non_peer_port_rejected():
    bad = VALID_PEER.replace("missing = []", 'missing = ["bogus"]')
    with pytest.raises(Exception, match="not a declared peer-link port"):
        _parse(bad)


# A valid role with the full ABI: tensor memref, dims (2) PID index args, then a stream per port.
VALID_ROLE_PORTS = """
module {
  func.func @pe(%a: memref<2x2xf32>, %pi: index, %pj: index, %e: !allo.stream<f32, 2>, %w: !allo.stream<f32, 2>) {
    return
  }
  func.func @top(%A: memref<2x2xf32>) {
    spmw.map (%A)
      topology = #spmw.topology<grid = [2, 2], dims = 2, links = [
        #spmw.peer_link<port = "east", map = affine_map<(i, j) -> (i, j + 1)>, peer = "west", depth = 2>,
        #spmw.peer_link<port = "west", map = affine_map<(i, j) -> (i, j - 1)>, peer = "east", depth = 2>
      ]>
      roles = [#spmw.role<unit = @pe, missing = [], ports = ["east", "west"]>]
      : memref<2x2xf32>
    return
  }
}
"""


def test_role_ports_round_trips():
    assert "ports = [" in str(_parse(VALID_ROLE_PORTS))


def test_role_ports_count_mismatch_rejected():
    # ports declares one port but the func has two stream parameters
    bad = VALID_ROLE_PORTS.replace('ports = ["east", "west"]', 'ports = ["east"]')
    with pytest.raises(Exception, match="more stream parameters than declared ports"):
        _parse(bad)


def test_role_ports_non_peer_rejected():
    bad = VALID_ROLE_PORTS.replace(
        'ports = ["east", "west"]', 'ports = ["east", "bogus"]'
    )
    with pytest.raises(Exception, match="not a declared peer-link port"):
        _parse(bad)


def test_role_ports_duplicate_rejected():
    bad = VALID_ROLE_PORTS.replace(
        'ports = ["east", "west"]', 'ports = ["east", "east"]'
    )
    with pytest.raises(Exception, match="is repeated"):
        _parse(bad)


def test_stream_role_without_ports_rejected():
    # a role func with stream parameters must declare its ports
    bad = VALID_ROLE_PORTS.replace(', ports = ["east", "west"]', "")
    with pytest.raises(Exception, match="does not declare its ports"):
        _parse(bad)


def test_role_stream_depth_mismatch_rejected():
    # the east stream is typed depth 4 but the "east" peer link declares depth 2
    bad = VALID_ROLE_PORTS.replace(
        "%e: !allo.stream<f32, 2>", "%e: !allo.stream<f32, 4>"
    )
    with pytest.raises(Exception, match="declares depth 2"):
        _parse(bad)


def test_role_wrong_index_arity_rejected():
    # a stream role must take exactly `dims` (2) PID index args; drop one
    bad = VALID_ROLE_PORTS.replace("%pi: index, %pj: index, ", "%pi: index, ")
    with pytest.raises(Exception, match="index parameters but the grid has 2"):
        _parse(bad)


def test_halo_stream_depth_mismatch_rejected():
    # the loader stream is typed depth 4 but its boundary port "west" declares depth 2
    bad = VALID_HALO.replace(
        "func.func @load_a(%a: memref<2x2xf32>, %p: index, %s: !allo.stream<f32, 2>)",
        "func.func @load_a(%a: memref<2x2xf32>, %p: index, %s: !allo.stream<f32, 4>)",
    )
    with pytest.raises(Exception, match="stream has depth 4 but port"):
        _parse(bad)


def test_key_endpoint_depth_mismatch_rejected():
    # src depth 2, sink depth 4 for the same key
    bad = VALID_KEY.replace(
        '#spmw.key_link<port = "in", key = "c", end = "sink", depth = 2>',
        '#spmw.key_link<port = "in", key = "c", end = "sink", depth = 4>',
    )
    with pytest.raises(Exception, match="mismatched depth"):
        _parse(bad)


def test_role_index_only_wrong_arity_rejected():
    # a role taking PID index params (but no streams) must take exactly `dims` of them
    bad = """
module {
  func.func @pe(%p: index) {
    return
  }
  func.func @top(%A: memref<2x2xf32>) {
    spmw.map (%A)
      topology = #spmw.topology<grid = [2, 2], dims = 2, links = []>
      roles = [#spmw.role<unit = @pe, missing = []>]
      : memref<2x2xf32>
    return
  }
}
"""
    with pytest.raises(Exception, match="index parameters but the grid has 2"):
        _parse(bad)


def test_role_stream_base_type_mismatch_rejected():
    # the "east" link declares element type f32 but the stream bound to it is i32
    bad = VALID_ROLE_PORTS.replace(
        'peer = "west", depth = 2>,',
        'peer = "west", depth = 2, type = f32>,',
    ).replace("%e: !allo.stream<f32, 2>", "%e: !allo.stream<i32, 2>")
    with pytest.raises(Exception, match="element type"):
        _parse(bad)
