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

#include <map>
#include <string>

using namespace mlir;
using namespace mlir::allo;

namespace {

int64_t floorDiv(int64_t a, int64_t b) {
  int64_t q = a / b, r = a % b;
  return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
}

/// Evaluate an affine expression (dims only) at a constant coordinate. Handles
/// the full affine grammar -- add, mul, and the semi-affine mod / floordiv /
/// ceildiv by a constant -- with MLIR's floor-based semantics, so a predicate
/// like `(i + j) mod 2` folds to a constant tag.
int64_t evalExpr(AffineExpr expr, ArrayRef<int64_t> coord) {
  if (auto c = llvm::dyn_cast<AffineConstantExpr>(expr))
    return c.getValue();
  if (auto d = llvm::dyn_cast<AffineDimExpr>(expr))
    return coord[d.getPosition()];
  auto bin = llvm::cast<AffineBinaryOpExpr>(expr);
  int64_t l = evalExpr(bin.getLHS(), coord), r = evalExpr(bin.getRHS(), coord);
  switch (bin.getKind()) {
  case AffineExprKind::Add:
    return l + r;
  case AffineExprKind::Mul:
    return l * r;
  case AffineExprKind::FloorDiv:
    return floorDiv(l, r);
  case AffineExprKind::CeilDiv:
    return -floorDiv(-l, r);
  case AffineExprKind::Mod:
    return l - r * floorDiv(l, r);
  default:
    return 0;
  }
}

/// Evaluate a static affine map (dims only, no symbols) at a constant
/// coordinate.
LogicalResult evalAffine(AffineMap map, ArrayRef<int64_t> coord,
                         SmallVectorImpl<int64_t> &out) {
  if (map.getNumSymbols() != 0 ||
      map.getNumDims() != static_cast<unsigned>(coord.size()))
    return failure();
  for (AffineExpr result : map.getResults())
    out.push_back(evalExpr(result, coord));
  return success();
}

bool inBounds(ArrayRef<int64_t> coord, ArrayRef<int64_t> grid) {
  for (auto [c, extent] : llvm::zip(coord, grid))
    if (c < 0 || c >= extent)
      return false;
  return true;
}

/// A role: the boundary ports whose absence selects it, and an optional
/// predicate (an affine map from the grid coordinate to an integer tag) that
/// further splits grid points sharing its link-presence class.
struct RoleInfo {
  SmallVector<StringRef> missing;
  AffineMap predicate;
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

  /// The index of the role that fits a grid point (most specific missing set
  /// wins), or -1 for none / an ambiguous tie. When several roles share the
  /// most-specific missing set they are compute variants distinguished by a
  /// coordinate predicate: a predicated role is eligible where its indicator is
  /// nonzero, and a role without a predicate is the default (chosen where no
  /// variant is eligible). This is how distinct predicate-selected bodies stay
  /// separate instances rather than being merged.
  int selectRole(ArrayRef<RoleInfo> roles, ArrayRef<StringRef> missing,
                 ArrayRef<int64_t> coord) {
    int bestSize = -1;
    SmallVector<int> fits;
    for (auto [i, role] : llvm::enumerate(roles)) {
      bool ok = llvm::all_of(role.missing, [&](StringRef port) {
        return llvm::is_contained(missing, port);
      });
      if (!ok)
        continue;
      int size = static_cast<int>(role.missing.size());
      if (size > bestSize) {
        bestSize = size;
        fits.assign(1, static_cast<int>(i));
      } else if (size == bestSize) {
        fits.push_back(static_cast<int>(i));
      }
    }
    if (fits.size() <= 1)
      return fits.empty() ? -1 : fits.front();
    int eligible = -1, eligibleCount = 0, dflt = -1, dfltCount = 0;
    for (int i : fits) {
      if (roles[i].predicate) {
        SmallVector<int64_t> tag;
        if (failed(evalAffine(roles[i].predicate, coord, tag)) || tag.empty())
          return -1;
        if (llvm::any_of(tag, [](int64_t t) { return t != 0; })) {
          eligible = i;
          ++eligibleCount;
        }
      } else {
        dflt = i;
        ++dfltCount;
      }
    }
    if (eligibleCount == 1)
      return eligible;
    if (eligibleCount == 0 && dfltCount == 1)
      return dflt;
    return -1;
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
      if (auto pred = role.getPredicate())
        info.predicate = pred.getAffineMap();
      roles.push_back(info);
    }

    int64_t total = 1;
    for (int64_t extent : grid)
      total *= extent;
    SmallVector<int64_t> counts(roles.size(), 0);
    // The link-presence classification is independent of the declared roles:
    // two grid points share a class when the same set of links is missing at
    // them. For a 2-D mesh (extents >= 3) this yields the nine classes
    // (interior, four edges, four corners), constant as the grid scales -- the
    // O(#roles) count the synthesis-time-win rests on. Keyed by the sorted
    // missing-port signature, std::map orders classes canonically.
    std::map<std::string, int64_t> linkClasses;

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
      int role = selectRole(roles, missing, coord);
      if (role < 0)
        return map.emitOpError("no unambiguous role fits a grid point");
      ++counts[role];

      SmallVector<StringRef> sig(missing.begin(), missing.end());
      llvm::sort(sig);
      std::string key;
      for (StringRef port : sig)
        key += (key.empty() ? "" : ",") + port.str();
      // A predicate on the selected role splits its link-presence class: two
      // points in the same class but with different predicate tags are distinct
      // instances. The key records the selected role's identity (its index) as
      // well as the tag values, so two *different* predicate roles that happen
      // to evaluate to the same tag do not collapse to one class.
      if (AffineMap pred = roles[role].predicate) {
        SmallVector<int64_t> tag;
        if (failed(evalAffine(pred, coord, tag)))
          return map.emitOpError("role predicate is not a static affine map");
        key += "#" + std::to_string(role);
        for (int64_t t : tag)
          key += ":" + std::to_string(t);
      }
      ++linkClasses[key];
    }

    OpBuilder builder(map);
    SmallVector<int64_t> classCounts;
    SmallVector<Attribute> classKeys;
    for (const auto &entry : linkClasses) {
      classCounts.push_back(entry.second);
      classKeys.push_back(builder.getStringAttr(entry.first));
    }

    map->setAttr("spmw.partition", builder.getDenseI64ArrayAttr(counts));
    // The number of link-presence classes (array length) is the O(#roles)
    // count; the entries are the per-class grid-point instance counts, which
    // scale with the grid.
    map->setAttr("spmw.link_classes",
                 builder.getDenseI64ArrayAttr(classCounts));
    // The identity of each class, parallel to spmw.link_classes: the sorted
    // missing-port signature with a `#<tag>` suffix per predicate tag. This
    // carries the (missing, predicate tag) identity into the IR so distinct
    // predicate-selected classes are not merged by downstream emission.
    map->setAttr("spmw.link_class_keys", builder.getArrayAttr(classKeys));
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
