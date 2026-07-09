/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===- SPMWUnroll.cpp - expand spmw.map over its grid -----------*- C++ -*-===//
//
// This pass is the structural consumer of the rolled spmw.map op. It replaces
// each spmw.map with one func.call per grid point: for every coordinate it
// evaluates the topology's affine peer links to find which local ports fall
// off the grid (the "missing" ports), selects the boundary role whose missing
// set matches, and emits a call to that role's func.func. The role func bodies
// stay O(#roles) -- they are never cloned per grid point -- while the call
// count is O(product(grid)). Call operands are built to match the callee's
// signature: the map's tensor operands feed the leading memref parameters,
// arith.constant indices feed the per-dimension PID parameters, and one
// allo.stream_construct per peer channel feeds the stream parameters (a channel
// is shared by the two neighbors it connects). For a port whose peer is
// off-grid, the map's spmw.halo tasks supply the missing endpoint: a loader
// produces into that boundary channel, or a drain consumes from it, so no
// off-grid channel is left with only one side.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"
#include "allo/Dialect/SPMW/SPMWAttrs.h"
#include "allo/Dialect/SPMW/SPMWOps.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <map>
#include <string>

using namespace mlir;
using namespace mlir::allo;

namespace {

/// Evaluate a static affine map (dims only, no symbols) at a constant
/// coordinate, appending each result to `out`. Fails if the map has symbols,
/// the dim count disagrees with the coordinate, or a result is non-constant.
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
    AffineExpr folded = result.replaceDims(repl);
    auto constant = llvm::dyn_cast<AffineConstantExpr>(folded);
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

/// Floor division matching MLIR affine semantics (rounds toward -inf).
int64_t predFloorDiv(int64_t a, int64_t b) {
  int64_t q = a / b, r = a % b;
  return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
}

/// Evaluate a dims-only affine predicate expression at a constant coordinate,
/// folding the semi-affine mod / floordiv / ceildiv (the peer-map `evalAffine`
/// above handles only translations, which never need folding). So an indicator
/// like `(d0 + d1) mod 2` collapses to a constant tag.
int64_t evalPredExpr(AffineExpr expr, ArrayRef<int64_t> coord) {
  if (auto c = llvm::dyn_cast<AffineConstantExpr>(expr))
    return c.getValue();
  if (auto d = llvm::dyn_cast<AffineDimExpr>(expr))
    return coord[d.getPosition()];
  auto bin = llvm::cast<AffineBinaryOpExpr>(expr);
  int64_t l = evalPredExpr(bin.getLHS(), coord);
  int64_t r = evalPredExpr(bin.getRHS(), coord);
  switch (bin.getKind()) {
  case AffineExprKind::Add:
    return l + r;
  case AffineExprKind::Mul:
    return l * r;
  case AffineExprKind::FloorDiv:
    return predFloorDiv(l, r);
  case AffineExprKind::CeilDiv:
    return -predFloorDiv(-l, r);
  case AffineExprKind::Mod:
    return l - r * predFloorDiv(l, r);
  default:
    return 0;
  }
}

/// Whether a role's predicate indicator is nonzero at a coordinate.
bool predicateHolds(AffineMap pred, ArrayRef<int64_t> coord) {
  for (AffineExpr result : pred.getResults())
    if (evalPredExpr(result, coord) != 0)
      return true;
  return false;
}

/// Row-major linear index of a coordinate within the grid.
int64_t linearIndex(ArrayRef<int64_t> coord, ArrayRef<int64_t> grid) {
  int64_t index = 0;
  for (auto [c, extent] : llvm::zip(coord, grid))
    index = index * extent + c;
  return index;
}

/// A role: the func it names, the boundary ports whose absence selects it, and
/// (optionally) the local port each of its stream parameters carries, in order.
struct RoleInfo {
  AffineMap predicate;
  FlatSymbolRefAttr unit;
  SmallVector<StringRef> missing;
  SmallVector<StringRef> ports;
};

/// One local peer port at a coordinate: whether its peer is on-grid, and if so
/// the peer's linear index and the peer-side port name (for channel sharing).
struct PortPeer {
  StringRef port;
  bool onGrid;
  int64_t peerIndex;
  StringRef peerPort;
};

struct SPMWUnrollPass
    : public mlir::allo::impl::SPMWUnrollBase<SPMWUnrollPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // The expansion creates func.call, arith.constant, and
    // allo.stream_construct ops, so those dialects must be loaded before the
    // pass runs.
    registry
        .insert<allo::AlloDialect, func::FuncDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<spmw::MapOp> maps;
    module.walk([&](spmw::MapOp op) { maps.push_back(op); });
    SymbolTable symbolTable(module);
    for (spmw::MapOp map : maps)
      if (failed(expand(map, symbolTable)))
        return signalPassFailure();
  }

  /// Select the role for a point given its missing ports and coordinate: a role
  /// fits when its missing set is a subset of the point's; the most specific
  /// (largest) wins. When several roles share the most-specific missing set
  /// they are predicate-selected compute variants -- a predicated role is
  /// eligible where its indicator is nonzero, and the unpredicated role is the
  /// default -- mirroring spmw-role-partition. Any other incomparable fit is a
  /// genuine ambiguity.
  LogicalResult selectRole(spmw::MapOp map, ArrayRef<RoleInfo> roles,
                           ArrayRef<StringRef> missing, ArrayRef<int64_t> coord,
                           const RoleInfo *&selected) {
    int bestSize = -1;
    SmallVector<const RoleInfo *> best;
    for (const RoleInfo &role : roles) {
      bool fits = llvm::all_of(role.missing, [&](StringRef port) {
        return llvm::is_contained(missing, port);
      });
      if (!fits)
        continue;
      int size = static_cast<int>(role.missing.size());
      if (size > bestSize) {
        bestSize = size;
        best.assign(1, &role);
      } else if (size == bestSize) {
        best.push_back(&role);
      }
    }
    if (best.empty())
      return map.emitOpError("no role fits a grid point missing ports");
    if (best.size() == 1) {
      selected = best.front();
      return success();
    }
    const RoleInfo *eligible = nullptr, *dflt = nullptr;
    int eligibleCount = 0, dfltCount = 0;
    for (const RoleInfo *role : best) {
      if (role->predicate) {
        if (predicateHolds(role->predicate, coord)) {
          eligible = role;
          ++eligibleCount;
        }
      } else {
        dflt = role;
        ++dfltCount;
      }
    }
    if (eligibleCount == 1) {
      selected = eligible;
      return success();
    }
    if (eligibleCount == 0 && dfltCount == 1) {
      selected = dflt;
      return success();
    }
    return map.emitOpError("ambiguous role for a grid point; declare a role "
                           "for that exact boundary");
  }

  LogicalResult expand(spmw::MapOp map, SymbolTable &symbolTable);
};

LogicalResult SPMWUnrollPass::expand(spmw::MapOp map,
                                     SymbolTable &symbolTable) {
  spmw::TopologyAttr topology = map.getTopology();
  ArrayRef<int64_t> grid = topology.getGrid();
  unsigned dims = static_cast<unsigned>(topology.getDims());

  // The peer links describe the interconnect pattern; each applies to every
  // coordinate. Their sorted local-port names are the canonical order the role
  // funcs list their stream parameters in.
  SmallVector<spmw::PeerLinkAttr> peers;
  for (Attribute link : topology.getLinks())
    if (auto peer = llvm::dyn_cast<spmw::PeerLinkAttr>(link))
      peers.push_back(peer);
  SmallVector<StringRef> sortedPorts;
  for (spmw::PeerLinkAttr peer : peers)
    sortedPorts.push_back(peer.getPort());
  llvm::sort(sortedPorts);

  // key_link channels are resolved by rendezvous key, not by grid neighbor, so
  // this grid-expansion pass cannot wire them. Fail closed rather than silently
  // drop them (channel-resolution will own key links).
  for (Attribute link : topology.getLinks())
    if (llvm::isa<spmw::KeyLinkAttr>(link))
      return map.emitOpError(
          "spmw-unroll does not yet lower key_link channels");

  SmallVector<RoleInfo> roles;
  for (Attribute roleAttr : map.getRoles()) {
    auto role = llvm::dyn_cast<spmw::RoleAttr>(roleAttr);
    if (!role)
      return map.emitOpError("each role must be an spmw.role attribute");
    RoleInfo info;
    info.unit = role.getUnit();
    if (auto pred = role.getPredicate())
      info.predicate = pred.getAffineMap();
    for (Attribute edge : role.getMissing())
      info.missing.push_back(llvm::cast<StringAttr>(edge).getValue());
    if (ArrayAttr ports = role.getPorts())
      for (Attribute port : ports)
        info.ports.push_back(llvm::cast<StringAttr>(port).getValue());
    roles.push_back(info);
  }

  OpBuilder builder(map);
  Location loc = map.getLoc();
  ValueRange tensors = map.getTensors();
  // Peer channels shared across neighbors, keyed by the canonical endpoint.
  std::map<std::string, Value> channels;

  int64_t total = 1;
  for (int64_t extent : grid)
    total *= extent;

  SmallVector<int64_t> coord(dims, 0);
  for (int64_t point = 0; point < total; ++point) {
    // Decode the row-major coordinate.
    int64_t rem = point;
    for (int d = static_cast<int>(dims) - 1; d >= 0; --d) {
      coord[d] = rem % grid[d];
      rem /= grid[d];
    }

    // Resolve each local port's peer and collect the missing ones.
    SmallVector<PortPeer> ports;
    SmallVector<StringRef> missing;
    for (spmw::PeerLinkAttr peer : peers) {
      SmallVector<int64_t> peerCoord;
      if (failed(
              evalAffine(peer.getPeerMap().getAffineMap(), coord, peerCoord)))
        return map.emitOpError("peer link '")
               << peer.getPort() << "' has a non-static map";
      PortPeer info;
      info.port = peer.getPort();
      info.onGrid = inBounds(peerCoord, grid);
      info.peerIndex = info.onGrid ? linearIndex(peerCoord, grid) : -1;
      info.peerPort = peer.getPeerPort();
      ports.push_back(info);
      if (!info.onGrid)
        missing.push_back(peer.getPort());
    }

    const RoleInfo *role = nullptr;
    if (failed(selectRole(map, roles, missing, coord, role)))
      return failure();
    auto callee = symbolTable.lookup<func::FuncOp>(role->unit.getValue());
    if (!callee)
      return map.emitOpError("role references undefined function '")
             << role->unit.getValue() << "'";

    // Build the call operands to match the callee's signature.
    SmallVector<Value> operands;
    unsigned tensorIdx = 0, pidIdx = 0, streamIdx = 0;
    for (Type input : callee.getFunctionType().getInputs()) {
      if (llvm::isa<MemRefType>(input)) {
        if (tensorIdx >= tensors.size())
          return map.emitOpError("role '")
                 << role->unit.getValue()
                 << "' needs more tensor operands than the map provides";
        operands.push_back(tensors[tensorIdx++]);
      } else if (llvm::isa<IndexType>(input)) {
        if (pidIdx >= dims)
          return map.emitOpError("role '")
                 << role->unit.getValue()
                 << "' has more index parameters than the grid has dimensions";
        operands.push_back(
            builder.create<arith::ConstantIndexOp>(loc, coord[pidIdx++]));
      } else if (llvm::isa<StreamType>(input)) {
        // A role's stream params bind to the local ports it declares (by name);
        // a role that declares no ports falls back to sorted local-port order.
        ArrayRef<StringRef> order = role->ports.empty()
                                        ? ArrayRef<StringRef>(sortedPorts)
                                        : ArrayRef<StringRef>(role->ports);
        if (streamIdx >= order.size())
          return map.emitOpError("role '")
                 << role->unit.getValue()
                 << "' has more stream parameters than declared ports";
        StringRef portName = order[streamIdx++];
        const PortPeer *info = nullptr;
        for (const PortPeer &candidate : ports)
          if (candidate.port == portName) {
            info = &candidate;
            break;
          }
        if (!info)
          return map.emitOpError("no peer link for port '") << portName << "'";
        int64_t self = linearIndex(coord, grid);
        std::string key;
        if (info->onGrid) {
          // Canonical undirected key: the smaller of the two endpoints, so the
          // neighbor resolves to the same allo.stream_construct.
          std::string a = std::to_string(self) + ":" + info->port.str();
          std::string b =
              std::to_string(info->peerIndex) + ":" + info->peerPort.str();
          key = std::min(a, b);
        } else {
          key = std::to_string(self) + ":" + info->port.str();
        }
        Value &channel = channels[key];
        if (!channel)
          channel = builder.create<allo::StreamConstructOp>(loc, input);
        operands.push_back(channel);
      } else {
        return map.emitOpError("role '")
               << role->unit.getValue()
               << "' has an unsupported parameter type";
      }
    }
    builder.create<func::CallOp>(loc, callee, operands);

    // Wire the halo loader/drain tasks for this point's missing (off-grid)
    // ports to the same boundary channels the compute call just created, so no
    // off-grid channel is left dangling.
    if (ArrayAttr haloTasks = map.getHaloAttr()) {
      int64_t self = linearIndex(coord, grid);
      for (Attribute haloAttr : haloTasks) {
        auto task = llvm::dyn_cast<spmw::HaloAttr>(haloAttr);
        if (!task || !llvm::is_contained(missing, task.getPort()))
          continue;
        auto haloFn =
            symbolTable.lookup<func::FuncOp>(task.getUnit().getValue());
        if (!haloFn)
          return map.emitOpError("halo task references undefined function '")
                 << task.getUnit().getValue() << "'";
        // The boundary channel must be the one the compute call keyed for this
        // off-grid port; if the selected compute role has no stream endpoint on
        // this port, the loader/drain would feed a disconnected FIFO -- error
        // rather than silently allocating a one-sided channel.
        std::string key = std::to_string(self) + ":" + task.getPort().str();
        auto found = channels.find(key);
        if (found == channels.end())
          return map.emitOpError("halo task for port '")
                 << task.getPort()
                 << "' has no matching compute-role stream endpoint at grid "
                    "point "
                 << self;
        Value channel = found->second;
        SmallVector<Value> haloOperands;
        if (task.getKind() == "load") {
          if (task.getOperand() < 0 ||
              task.getOperand() >= static_cast<int64_t>(tensors.size()))
            return map.emitOpError("halo operand out of range");
          haloOperands.push_back(tensors[task.getOperand()]);
          haloOperands.push_back(builder.create<arith::ConstantIndexOp>(
              loc, coord[task.getAxis()]));
        }
        haloOperands.push_back(channel);
        builder.create<func::CallOp>(loc, haloFn, haloOperands);
      }
    }
  }

  map.erase();
  return success();
}

} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createSPMWUnrollPass() {
  return std::make_unique<SPMWUnrollPass>();
}

} // namespace allo
} // namespace mlir
