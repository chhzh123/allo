/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===- SPMWResolveChannels.cpp - group peer links into families -*- C++ -*-===//
//
// This pass resolves a spmw.map's peer interconnect into distinct channel
// families. A peer link and its reciprocal are one undirected channel, named by
// the unordered pair of ports they connect (e.g. "east/west"); links sharing
// that pair form one family -- the FIFO array an HLS emitter declares. The
// sorted family names are attached to the map as a `spmw.channel_families`
// array attribute. The family count is constant as the grid scales even though
// the channel instance count is O(product(grid)), so this is the compact
// interconnect representation the rolled HLS emitter consumes.
//
// When the map is folded, a family whose forwarding channel spans a folded axis
// is time-multiplexed onto one physical link in an order a linear FIFO cannot
// preserve, so it is reclassified into `spmw.buffer_families` (an addressed
// buffer) instead of remaining a stream. Families the fold does not affect stay
// in `spmw.channel_families`, so an unfolded map is byte-identical.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "allo/Dialect/SPMW/SPMWAttrs.h"
#include "allo/Dialect/SPMW/SPMWOps.h"
#include "allo/Transforms/Passes.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include "llvm/ADT/Twine.h"
#include <map>
#include <set>

#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::allo;

namespace {

// The grid axes a peer link's affine map displaces: the result positions that
// are not the identity dimension (e.g. east's (i,j)->(i,j+1) displaces axis 1).
// A family's connection axes are the axes its forwarding channel spans.
static void collectDisplacedAxes(AffineMap m, std::set<int64_t> &axes) {
  MLIRContext *ctx = m.getContext();
  for (unsigned i = 0, e = m.getNumResults(); i < e; ++i)
    if (m.getResult(i) != getAffineDimExpr(i, ctx))
      axes.insert(static_cast<int64_t>(i));
}

struct SPMWResolveChannelsPass
    : public mlir::allo::impl::SPMWResolveChannelsBase<
          SPMWResolveChannelsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<spmw::MapOp> maps;
    module.walk([&](spmw::MapOp op) { maps.push_back(op); });
    for (spmw::MapOp map : maps)
      if (failed(resolve(map)))
        return signalPassFailure();
  }

  LogicalResult resolve(spmw::MapOp map) {
    // Each family maps to its FIFO depth (the peer link's depth); a link and
    // its reciprocal, and both endpoints of the family, must agree on that
    // depth. familyAxes records the grid axes each family's channel spans.
    std::map<std::string, int64_t> familyDepth;
    std::map<std::string, std::set<int64_t>> familyAxes;
    // Key (rendezvous) families group by their key string -- the channel-group
    // family, one FIFO array -- with a consistent per-family depth.
    std::map<std::string, int64_t> keyFamilyDepth;
    for (Attribute link : map.getTopology().getLinks()) {
      auto peer = llvm::dyn_cast<spmw::PeerLinkAttr>(link);
      if (!peer) {
        // A key_link rendezvous by key: group it under its key string. The
        // precise per-instance src/sink matching is a frontend concern; here
        // the family is the rolled interconnect unit the emitter declares.
        if (auto key = llvm::dyn_cast<spmw::KeyLinkAttr>(link)) {
          auto family = llvm::dyn_cast<StringAttr>(key.getKey());
          if (!family)
            return map.emitOpError(
                "spmw-resolve-channels needs a string key on a key_link");
          std::string name = family.getValue().str();
          auto it = keyFamilyDepth.find(name);
          if (it == keyFamilyDepth.end())
            keyFamilyDepth[name] = key.getDepth();
          else if (it->second != key.getDepth())
            return map.emitOpError("key channel family '")
                   << name << "' has links with mismatched depth";
        } else if (auto edge = llvm::dyn_cast<spmw::EdgeLinkAttr>(link)) {
          // An explicit edge groups into a family by its own per-edge unordered
          // {port, peerPort} pair -- the tree's right-child `up` (peer `right`)
          // joins the parent's `right` (peer `up`) family, distinct from a
          // left-child `up` (peer `left`) -- so the true per-edge peer port
          // drives the channel family.
          StringRef ea = edge.getPort(), eb = edge.getPeerPort();
          std::string family = (ea <= eb ? (llvm::Twine(ea) + "/" + eb)
                                         : (llvm::Twine(eb) + "/" + ea))
                                   .str();
          auto it = familyDepth.find(family);
          if (it == familyDepth.end())
            familyDepth[family] = edge.getDepth();
          else if (it->second != edge.getDepth())
            return map.emitOpError("channel family '")
                   << family << "' has links with mismatched depth";
        }
        continue;
      }
      // The unordered pair {port, peerPort} names the family, so a link and its
      // reciprocal collapse to one entry.
      StringRef a = peer.getPort(), b = peer.getPeerPort();
      std::string family =
          (a <= b ? (llvm::Twine(a) + "/" + b) : (llvm::Twine(b) + "/" + a))
              .str();
      auto it = familyDepth.find(family);
      if (it == familyDepth.end())
        familyDepth[family] = peer.getDepth();
      else if (it->second != peer.getDepth())
        return map.emitOpError("channel family '")
               << family << "' has links with mismatched depth";
      collectDisplacedAxes(peer.getPeerMap().getAffineMap(),
                           familyAxes[family]);
    }

    // A fold time-multiplexes the map: a factor > 1 along a family's connection
    // axis makes one physical link carry several logical instances of that
    // forwarding channel in an order a linear FIFO cannot preserve, so the
    // family must become an addressed buffer. A fold only along axes the family
    // does not span keeps FIFO order (block/cyclic time-muxing along an
    // unrelated axis is still ordered), so it stays a stream.
    ArrayRef<int64_t> fold;
    if (DenseI64ArrayAttr foldAttr = map.getFoldAttr())
      fold = foldAttr.asArrayRef();

    // std::map orders families canonically; emit stream families (and their
    // per-family FIFO depths) in that order -- the FIFO arrays the emitter
    // declares -- and any buffer families separately.
    SmallVector<StringRef> streamRefs, bufferRefs;
    SmallVector<int64_t> streamDepths, bufferDepths;
    for (const auto &entry : familyDepth) {
      bool isBuffer = false;
      for (int64_t axis : familyAxes[entry.first])
        if (axis < static_cast<int64_t>(fold.size()) && fold[axis] > 1) {
          isBuffer = true;
          break;
        }
      if (isBuffer) {
        bufferRefs.push_back(entry.first);
        bufferDepths.push_back(entry.second);
      } else {
        streamRefs.push_back(entry.first);
        streamDepths.push_back(entry.second);
      }
    }

    // Key (rendezvous) families. A fold time-multiplexes several logical PEs
    // onto one physical PE, which then random-accesses its lane family (the
    // butterfly permutation {i, i^(1<<s)}) instead of draining it in FIFO order
    // -- so a folded key family becomes an addressed buffer (it still needs
    // conflict-free banking, derived downstream from the access set). Unfolded,
    // it stays a FIFO stream. This is conservative: any fold reclassifies (the
    // FFT folds the butterfly axis, the case that matters), and a buffer still
    // serves FIFO access, so over-classifying is safe.
    bool folded = false;
    for (int64_t f : fold)
      if (f > 1) {
        folded = true;
        break;
      }
    for (const auto &entry : keyFamilyDepth) {
      if (folded) {
        bufferRefs.push_back(entry.first);
        bufferDepths.push_back(entry.second);
      } else {
        streamRefs.push_back(entry.first);
        streamDepths.push_back(entry.second);
      }
    }

    OpBuilder builder(map);
    map->setAttr("spmw.channel_families", builder.getStrArrayAttr(streamRefs));
    map->setAttr("spmw.channel_family_depths",
                 builder.getDenseI64ArrayAttr(streamDepths));
    // Emit the buffer attrs only when a fold reclassified something, so an
    // unfolded map's attributes are byte-identical to before.
    if (!bufferRefs.empty()) {
      map->setAttr("spmw.buffer_families", builder.getStrArrayAttr(bufferRefs));
      map->setAttr("spmw.buffer_family_depths",
                   builder.getDenseI64ArrayAttr(bufferDepths));
    }
    return success();
  }
};

} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createSPMWResolveChannelsPass() {
  return std::make_unique<SPMWResolveChannelsPass>();
}

} // namespace allo
} // namespace mlir
