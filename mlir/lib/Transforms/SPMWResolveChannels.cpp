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
    module.walk([&](spmw::MapOp map) { resolve(map); });
  }

  void resolve(spmw::MapOp map) {
    llvm::StringSet<> seen;
    SmallVector<std::string> families;
    for (Attribute link : map.getTopology().getLinks()) {
      auto peer = llvm::dyn_cast<spmw::PeerLinkAttr>(link);
      if (!peer)
        continue;
      // The unordered pair {port, peerPort} names the family, so a link and its
      // reciprocal collapse to one entry.
      StringRef a = peer.getPort(), b = peer.getPeerPort();
      std::string family =
          (a <= b ? (llvm::Twine(a) + "/" + b) : (llvm::Twine(b) + "/" + a))
              .str();
      if (seen.insert(family).second)
        families.push_back(family);
    }
    llvm::sort(families);
    SmallVector<StringRef> refs(families.begin(), families.end());
    OpBuilder builder(map);
    map->setAttr("spmw.channel_families", builder.getStrArrayAttr(refs));
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
