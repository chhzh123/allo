/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===- SPMWRolePartition.cpp - classify grid points by role -----*- C++ -*-===//
//
// This pass makes the O(#roles) grid-point classification of a spmw.map
// explicit in the IR. For each grid point it evaluates the topology's affine
// peer links to find the missing (off-grid) ports, selects the boundary role by
// the same subset rule spmw-unroll uses, and tallies how many grid points each
// role owns. The per-role counts (in role order) are attached to the map as a
// `spmw.partition` DenseI64ArrayAttr, so a downstream HLS emitter can emit one
// body per role and instantiate it over its point count -- the
// synthesis-time-win representation (a constant number of role bodies as the
// grid scales).
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "allo/Dialect/SPMW/SPMWAttrs.h"
#include "allo/Dialect/SPMW/SPMWOps.h"
#include "allo/Transforms/Passes.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::allo;

namespace {

/// Evaluate a static affine map (dims only, no symbols) at a constant
/// coordinate.
LogicalResult evalAffine(AffineMap map, ArrayRef<int64_t> coord,
                         SmallVectorImpl<int64_t> &out) {
  if (map.getNumSymbols() != 0 ||
      map.getNumDims() != static_cast<unsigned>(coord.size()))
    return failure();
  MLIRContext *ctx = map.getContext();
  SmallVector<AffineExpr> repl;
  for (int64_t c : coord)
    repl.push_back(getAffineConstantExpr(c, ctx));
  for (AffineExpr result : map.getResults()) {
    auto constant =
        llvm::dyn_cast<AffineConstantExpr>(result.replaceDims(repl));
    if (!constant)
      return failure();
    out.push_back(constant.getValue());
  }
  return success();
}

bool inBounds(ArrayRef<int64_t> coord, ArrayRef<int64_t> grid) {
  for (auto [c, extent] : llvm::zip(coord, grid))
    if (c < 0 || c >= extent)
      return false;
  return true;
}

/// A role: the boundary ports whose absence selects it.
struct RoleInfo {
  SmallVector<StringRef> missing;
};

struct SPMWRolePartitionPass
    : public mlir::allo::impl::SPMWRolePartitionBase<SPMWRolePartitionPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<spmw::MapOp> maps;
    module.walk([&](spmw::MapOp op) { maps.push_back(op); });
    for (spmw::MapOp map : maps)
      if (failed(partition(map)))
        return signalPassFailure();
  }

  /// The index of the role that fits a point's missing ports (most specific
  /// wins), or -1 for none / an ambiguous tie.
  int selectRole(ArrayRef<RoleInfo> roles, ArrayRef<StringRef> missing) {
    int bestSize = -1, best = -1, ties = 0;
    for (auto [i, role] : llvm::enumerate(roles)) {
      bool fits = llvm::all_of(role.missing, [&](StringRef port) {
        return llvm::is_contained(missing, port);
      });
      if (!fits)
        continue;
      int size = static_cast<int>(role.missing.size());
      if (size > bestSize) {
        bestSize = size;
        best = static_cast<int>(i);
        ties = 1;
      } else if (size == bestSize) {
        ++ties;
      }
    }
    return ties == 1 ? best : -1;
  }

  LogicalResult partition(spmw::MapOp map) {
    spmw::TopologyAttr topology = map.getTopology();
    ArrayRef<int64_t> grid = topology.getGrid();
    unsigned dims = static_cast<unsigned>(topology.getDims());

    SmallVector<spmw::PeerLinkAttr> peers;
    for (Attribute link : topology.getLinks())
      if (auto peer = llvm::dyn_cast<spmw::PeerLinkAttr>(link))
        peers.push_back(peer);

    SmallVector<RoleInfo> roles;
    for (Attribute roleAttr : map.getRoles()) {
      auto role = llvm::dyn_cast<spmw::RoleAttr>(roleAttr);
      if (!role)
        return map.emitOpError("each role must be an spmw.role attribute");
      RoleInfo info;
      for (Attribute edge : role.getMissing())
        info.missing.push_back(llvm::cast<StringAttr>(edge).getValue());
      roles.push_back(info);
    }

    int64_t total = 1;
    for (int64_t extent : grid)
      total *= extent;
    SmallVector<int64_t> counts(roles.size(), 0);

    SmallVector<int64_t> coord(dims, 0);
    for (int64_t point = 0; point < total; ++point) {
      int64_t rem = point;
      for (int d = static_cast<int>(dims) - 1; d >= 0; --d) {
        coord[d] = rem % grid[d];
        rem /= grid[d];
      }
      SmallVector<StringRef> missing;
      for (spmw::PeerLinkAttr peer : peers) {
        SmallVector<int64_t> peerCoord;
        if (failed(
                evalAffine(peer.getPeerMap().getAffineMap(), coord, peerCoord)))
          return map.emitOpError("peer link '")
                 << peer.getPort() << "' has a non-static map";
        if (!inBounds(peerCoord, grid))
          missing.push_back(peer.getPort());
      }
      int role = selectRole(roles, missing);
      if (role < 0)
        return map.emitOpError("no unambiguous role fits a grid point");
      ++counts[role];
    }

    OpBuilder builder(map);
    map->setAttr("spmw.partition", builder.getDenseI64ArrayAttr(counts));
    return success();
  }
};

} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createSPMWRolePartitionPass() {
  return std::make_unique<SPMWRolePartitionPass>();
}

} // namespace allo
} // namespace mlir
