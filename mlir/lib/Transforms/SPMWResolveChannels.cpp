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
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "allo/Dialect/SPMW/SPMWAttrs.h"
#include "allo/Dialect/SPMW/SPMWOps.h"
#include "allo/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include "llvm/ADT/Twine.h"
#include <map>

#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::allo;

namespace {

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
    // depth.
    std::map<std::string, int64_t> familyDepth;
    for (Attribute link : map.getTopology().getLinks()) {
      auto peer = llvm::dyn_cast<spmw::PeerLinkAttr>(link);
      if (!peer) {
        // key_link channels rendezvous by key, not by grid-neighbor port pair,
        // so this peer-family resolver cannot group them. Fail closed rather
        // than silently omit them from spmw.channel_families.
        if (llvm::isa<spmw::KeyLinkAttr>(link))
          return map.emitOpError(
              "spmw-resolve-channels does not yet resolve key_link channels");
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
    }
    // std::map orders families canonically; emit the family names and the
    // per-family FIFO depths in that order (the FIFO arrays the emitter
    // declares).
    SmallVector<StringRef> refs;
    SmallVector<int64_t> depths;
    for (const auto &entry : familyDepth) {
      refs.push_back(entry.first);
      depths.push_back(entry.second);
    }
    OpBuilder builder(map);
    map->setAttr("spmw.channel_families", builder.getStrArrayAttr(refs));
    map->setAttr("spmw.channel_family_depths",
                 builder.getDenseI64ArrayAttr(depths));
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
