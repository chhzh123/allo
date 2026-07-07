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


def test_spmw_rank_round_trips():
    ir = """
    module {
      func.func @unit() {
        %0:2 = "spmw.rank"() : () -> (index, index)
        return
      }
    }
    """
    ctx = _context()
    with ctx, Location.unknown():
        module = Module.parse(ir)
    assert "spmw.rank" in str(module)


def test_spmw_map_round_trips_and_reparses():
    # Parse the generic form (always valid for a registered, verifying op), then
    # round-trip the pretty-printed custom form back through the parser.
    generic = """
    module {
      func.func @pe_interior(%a: memref<2x2xf32>) {
        return
      }
      func.func @top(%A: memref<2x2xf32>, %B: memref<2x2xf32>) {
        "spmw.map"(%A, %B) {
          grid = array<i64: 2, 2>,
          topologyDims = 2 : i64,
          links = [affine_map<(i, j) -> (i, j)>],
          roles = [@pe_interior]
        } : (memref<2x2xf32>, memref<2x2xf32>) -> ()
        return
      }
    }
    """
    ctx = _context()
    with ctx, Location.unknown():
        module = Module.parse(generic)
        printed = str(module)
        assert "spmw.map" in printed
        reparsed = Module.parse(printed)
    assert "spmw.map" in str(reparsed)


def test_spmw_map_grid_rank_mismatch_rejected():
    # topology dims (3) disagree with the grid rank (2): the verifier must reject it.
    bad = """
    module {
      func.func @pe(%a: memref<2x2xf32>) {
        return
      }
      func.func @top(%A: memref<2x2xf32>) {
        "spmw.map"(%A) {
          grid = array<i64: 2, 2>,
          topologyDims = 3 : i64,
          links = [],
          roles = [@pe]
        } : (memref<2x2xf32>) -> ()
        return
      }
    }
    """
    ctx = _context()
    with ctx, Location.unknown():
        with pytest.raises(Exception):
            Module.parse(bad)
