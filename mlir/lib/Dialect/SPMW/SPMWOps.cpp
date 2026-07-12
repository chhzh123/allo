/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===- SPMWOps.cpp - spmw dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "allo/Dialect/SPMW/SPMWOps.h"
#include "allo/Dialect/AlloTypes.h"
#include "allo/Dialect/SPMW/SPMWAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::spmw;

#define GET_OP_CLASSES
#include "allo/Dialect/SPMW/SPMWOps.cpp.inc"

namespace mlir {
namespace spmw {

// Whether an affine map is a pure per-dim translation ``(d0, d1, ...) -> (d0 +
// c0, d1 + c1, ...)`` -- the mesh/ring peer links whose reciprocal is a
// well-defined inverse. A general-affine peer map (e.g. the heap reduction
// tree's
// ``(d0 - 1) floordiv 2`` / ``d0 * 2 + 1``) is not a translation; it is an
// explicit, possibly non-reciprocal edge whose only role-partition use is the
// in-bounds check of its evaluated peer coordinate.
static bool isTranslationMap(AffineMap map) {
  if (map.getNumSymbols() != 0 || map.getNumResults() != map.getNumDims())
    return false;
  for (unsigned i = 0, e = map.getNumResults(); i < e; ++i) {
    AffineExpr expr = map.getResult(i);
    if (auto dim = llvm::dyn_cast<AffineDimExpr>(expr)) {
      if (dim.getPosition() != i)
        return false;
    } else if (auto bin = llvm::dyn_cast<AffineBinaryOpExpr>(expr)) {
      auto dim = llvm::dyn_cast<AffineDimExpr>(bin.getLHS());
      auto cst = llvm::dyn_cast<AffineConstantExpr>(bin.getRHS());
      if (bin.getKind() != AffineExprKind::Add || !dim || !cst ||
          dim.getPosition() != i)
        return false;
    } else {
      return false;
    }
  }
  return true;
}

LogicalResult MapOp::verify() {
  TopologyAttr topology = getTopology();
  int64_t dims = topology.getDims();
  ArrayRef<int64_t> grid = topology.getGrid();
  if (static_cast<int64_t>(grid.size()) != dims)
    return emitOpError("grid rank (")
           << grid.size() << ") does not match topology dims (" << dims << ")";
  for (int64_t extent : grid)
    if (extent <= 0)
      return emitOpError("grid extent must be positive; got ") << extent;

  // Each channel key (family) must have at least one source and one sink.
  llvm::DenseMap<Attribute, std::pair<int, int>> keyCounts;
  // The src and sink sharing a key must agree on depth and element type.
  llvm::DenseMap<Attribute, std::pair<int64_t, Type>> keyAbi;
  llvm::StringMap<PeerLinkAttr> peerByPort;
  for (Attribute linkAttr : topology.getLinks()) {
    if (auto peer = llvm::dyn_cast<PeerLinkAttr>(linkAttr)) {
      AffineMap map = peer.getPeerMap().getAffineMap();
      if (static_cast<int64_t>(map.getNumDims()) != dims ||
          static_cast<int64_t>(map.getNumResults()) != dims)
        return emitOpError("peer link '")
               << peer.getPort() << "' map rank does not match topology dims ("
               << dims << ")";
      if (peer.getDepth() <= 0)
        return emitOpError("peer link '")
               << peer.getPort() << "' has non-positive depth";
      if (!peerByPort.insert({peer.getPort(), peer}).second)
        return emitOpError("duplicate peer link port '")
               << peer.getPort() << "'";
    } else if (auto key = llvm::dyn_cast<KeyLinkAttr>(linkAttr)) {
      StringRef endpoint = key.getEndpoint();
      if (endpoint != "src" && endpoint != "sink")
        return emitOpError("key link '")
               << key.getPort() << "' endpoint must be \"src\" or \"sink\"";
      if (key.getDepth() <= 0)
        return emitOpError("key link '")
               << key.getPort() << "' has non-positive depth";
      std::pair<int, int> &count = keyCounts[key.getKey()];
      if (endpoint == "src")
        ++count.first;
      else
        ++count.second;
      std::pair<int64_t, Type> abi = {
          key.getDepth(),
          key.getElementType() ? key.getElementType().getValue() : Type()};
      auto insertion = keyAbi.try_emplace(key.getKey(), abi);
      if (!insertion.second) {
        std::pair<int64_t, Type> &seen = insertion.first->second;
        if (seen.first != abi.first)
          return emitOpError("key '")
                 << key.getKey() << "' endpoints have mismatched depth";
        if (seen.second && abi.second && seen.second != abi.second)
          return emitOpError("key '")
                 << key.getKey() << "' endpoints have mismatched element type";
      }
    } else {
      return emitOpError(
          "topology link must be a peer_link or key_link attribute");
    }
  }
  // A channel key names a channel family: 1->1 (peer), 1->N (scatter), N->1
  // (gather), or an N->N permutation (an FFT lane family is one FIFO array of
  // many 1->1 slots). Precise per-instance cardinality is a frontend concern
  // (key_channel_roles over the concrete keys); the rolled family must only
  // have a producer and a consumer, so a src-only or sink-only (dangling) key
  // fails -- UNLESS the missing endpoint is supplied externally by a region
  // stream (a scatter's source / a gather's sink), which the frontend records
  // on the op via the `spmw.external_srcs` / `spmw.external_sinks` discardable
  // attributes (families whose src / sink is an external stream_in /
  // stream_out).
  llvm::StringSet<> externalSrc, externalSink;
  if (auto arr = (*this)->getAttrOfType<ArrayAttr>("spmw.external_srcs"))
    for (Attribute a : arr)
      if (auto s = llvm::dyn_cast<StringAttr>(a))
        externalSrc.insert(s.getValue());
  if (auto arr = (*this)->getAttrOfType<ArrayAttr>("spmw.external_sinks"))
    for (Attribute a : arr)
      if (auto s = llvm::dyn_cast<StringAttr>(a))
        externalSink.insert(s.getValue());
  for (const auto &entry : keyCounts) {
    StringRef family;
    if (auto s = llvm::dyn_cast<StringAttr>(entry.first))
      family = s.getValue();
    int numSrc = entry.second.first + (externalSrc.contains(family) ? 1 : 0);
    int numSink = entry.second.second + (externalSink.contains(family) ? 1 : 0);
    if (numSrc < 1 || numSink < 1)
      return emitOpError(
                 "channel key must have at least one src and one sink; got ")
             << numSrc << " src and " << numSink << " sink";
  }

  // Peer links must be reciprocal: the link reached by following `port` must
  // point back on `peerPort` with an inverse map and matching depth. The
  // unroll pass shares one FIFO between the two endpoints and relies on this.
  AffineMap identity = AffineMap::getMultiDimIdentityMap(
      static_cast<unsigned>(dims), getContext());
  for (const auto &entry : peerByPort) {
    PeerLinkAttr link = entry.second;
    // A non-translation peer map (e.g. the heap tree's `(d0-1) floordiv 2`) is
    // an explicit, possibly non-reciprocal edge: its reciprocal is
    // coordinate-dependent (the tree's `up` peers a parent's `left` or `right`
    // by child parity), so the strict inverse-composition reciprocity below
    // does not apply. The role-partition pass classifies it purely by its
    // evaluated peer coordinate's in-bounds-ness.
    if (!isTranslationMap(link.getPeerMap().getAffineMap()))
      continue;
    auto peerIt = peerByPort.find(link.getPeerPort());
    if (peerIt == peerByPort.end())
      return emitOpError("peer link '")
             << link.getPort() << "' has no reciprocal link on peer port '"
             << link.getPeerPort() << "'";
    PeerLinkAttr back = peerIt->second;
    if (back.getPeerPort() != link.getPort())
      return emitOpError("peer link '")
             << link.getPort() << "' and '" << link.getPeerPort()
             << "' are not reciprocal";
    if (back.getPeerMap().getAffineMap().compose(
            link.getPeerMap().getAffineMap()) != identity)
      return emitOpError("peer link '")
             << link.getPort()
             << "' and its reciprocal maps do not compose to the identity";
    if (back.getDepth() != link.getDepth())
      return emitOpError("peer link '")
             << link.getPort() << "' and its reciprocal have mismatched depth";
    // Both endpoints share one FIFO, so they must agree on the element type.
    if (back.getElementType() != link.getElementType())
      return emitOpError("peer link '")
             << link.getPort()
             << "' and its reciprocal have mismatched element type";
  }

  for (Attribute roleAttr : getRoles()) {
    auto role = llvm::dyn_cast<RoleAttr>(roleAttr);
    if (!role)
      return emitOpError("each role must be an spmw.role attribute");
    llvm::StringSet<> seenMissing;
    for (Attribute missing : role.getMissing()) {
      auto port = llvm::dyn_cast<StringAttr>(missing);
      if (!port)
        return emitOpError("role missing entry must be a string");
      if (!peerByPort.contains(port.getValue()))
        return emitOpError("role missing port '")
               << port.getValue() << "' is not a declared peer-link port";
      if (!seenMissing.insert(port.getValue()).second)
        return emitOpError("role missing port '")
               << port.getValue() << "' is repeated";
    }
    if (ArrayAttr ports = role.getPorts()) {
      llvm::StringSet<> seenPorts;
      for (Attribute portAttr : ports) {
        auto port = llvm::dyn_cast<StringAttr>(portAttr);
        if (!port)
          return emitOpError("role port entry must be a string");
        if (!peerByPort.contains(port.getValue()))
          return emitOpError("role port '")
                 << port.getValue() << "' is not a declared peer-link port";
        if (!seenPorts.insert(port.getValue()).second)
          return emitOpError("role port '")
                 << port.getValue() << "' is repeated";
      }
    }
  }

  if (ArrayAttr halo = getHaloAttr()) {
    int64_t numTensors = static_cast<int64_t>(getTensors().size());
    llvm::StringSet<> seenHalo;
    for (Attribute haloAttr : halo) {
      auto task = llvm::dyn_cast<HaloAttr>(haloAttr);
      if (!task)
        return emitOpError("each halo task must be an spmw.halo attribute");
      if (task.getKind() != "load" && task.getKind() != "drain")
        return emitOpError("halo kind must be \"load\" or \"drain\"");
      if (task.getAxis() < 0 || task.getAxis() >= dims)
        return emitOpError("halo axis (")
               << task.getAxis() << ") out of range for topology dims (" << dims
               << ")";
      if (task.getKind() == "load" &&
          (task.getOperand() < 0 || task.getOperand() >= numTensors))
        return emitOpError("halo operand (")
               << task.getOperand() << ") out of range for " << numTensors
               << " tensor operands";
      // A halo task attaches where its port is off-grid, so the port must be a
      // real peer-link port of the topology.
      if (!peerByPort.contains(task.getPort()))
        return emitOpError("halo port '")
               << task.getPort() << "' is not a declared peer-link port";
      // At most one loader and one drain per port: two would create multiple
      // producers/consumers on the same boundary channel.
      std::string tag =
          (llvm::Twine(task.getPort()) + ":" + task.getKind()).str();
      if (!seenHalo.insert(tag).second)
        return emitOpError("duplicate halo task for port '")
               << task.getPort() << "' kind '" << task.getKind() << "'";
    }
  }

  if (DenseI64ArrayAttr fold = getFoldAttr())
    if (static_cast<int64_t>(fold.size()) != dims)
      return emitOpError("fold rank does not match topology dims (")
             << dims << ")";
  if (DenseI64ArrayAttr unroll = getUnrollAttr())
    if (static_cast<int64_t>(unroll.size()) != dims)
      return emitOpError("unroll rank does not match topology dims (")
             << dims << ")";

  return success();
}

LogicalResult MapOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  TypeRange tensorTypes = getTensors().getTypes();
  TopologyAttr topology = getTopology();
  int64_t dims = topology.getDims();
  llvm::StringMap<std::pair<int64_t, Type>> portAbi;
  for (Attribute linkAttr : topology.getLinks())
    if (auto peer = llvm::dyn_cast<PeerLinkAttr>(linkAttr))
      portAbi[peer.getPort()] = {
          peer.getDepth(),
          peer.getElementType() ? peer.getElementType().getValue() : Type()};
  for (Attribute roleAttr : getRoles()) {
    auto role = llvm::dyn_cast<RoleAttr>(roleAttr);
    if (!role)
      continue; // payload kind is checked by verify()
    auto fn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this,
                                                                role.getUnit());
    if (!fn)
      return emitOpError("role references undefined function '")
             << role.getUnit().getValue() << "'";
    // A role func's parameters are the map's tensor memrefs, then exactly
    // `dims` index PID parameters, then one stream per declared port -- in that
    // order.
    ArrayAttr ports = role.getPorts();
    StringRef unit = role.getUnit().getValue();
    unsigned ti = 0, idxCount = 0, streamIdx = 0;
    bool sawIndex = false, sawStream = false;
    for (Type input : fn.getFunctionType().getInputs()) {
      if (llvm::isa<MemRefType>(input)) {
        if (sawIndex || sawStream)
          return emitOpError("role '")
                 << unit
                 << "' has a memref argument after a per-instantiation "
                    "parameter; a role lists tensors, then PID indices, then "
                    "streams";
        if (ti >= tensorTypes.size())
          return emitOpError("role '")
                 << unit
                 << "' takes more memref arguments than the map has tensor "
                    "operands";
        if (input != tensorTypes[ti])
          return emitOpError("role '")
                 << unit << "' memref argument " << ti << " (" << input
                 << ") does not match map tensor operand " << ti << " ("
                 << tensorTypes[ti] << ")";
        ++ti;
      } else if (llvm::isa<IndexType>(input)) {
        if (sawStream)
          return emitOpError("role '")
                 << unit << "' has an index parameter after a stream parameter";
        sawIndex = true;
        ++idxCount;
      } else if (auto stream = llvm::dyn_cast<allo::StreamType>(input)) {
        sawStream = true;
        if (!ports)
          return emitOpError("role '")
                 << unit
                 << "' has stream parameters but does not declare its ports";
        if (streamIdx >= ports.size())
          return emitOpError("role '")
                 << unit << "' has more stream parameters than declared ports";
        StringRef portName =
            llvm::cast<StringAttr>(ports[streamIdx]).getValue();
        auto abiIt = portAbi.find(portName);
        if (abiIt != portAbi.end()) {
          if (static_cast<int64_t>(stream.getDepth()) != abiIt->second.first)
            return emitOpError("role '")
                   << unit << "' stream parameter " << streamIdx
                   << " has depth " << stream.getDepth() << " but port '"
                   << portName << "' declares depth " << abiIt->second.first;
          if (abiIt->second.second &&
              stream.getBaseType() != abiIt->second.second)
            return emitOpError("role '")
                   << unit << "' stream parameter " << streamIdx
                   << " has element type " << stream.getBaseType()
                   << " but port '" << portName << "' declares "
                   << abiIt->second.second;
        }
        ++streamIdx;
      } else {
        return emitOpError("role '")
               << unit << "' has an unsupported parameter type";
      }
    }
    if (ports && streamIdx != static_cast<unsigned>(ports.size()))
      return emitOpError("role '") << unit << "' declares " << ports.size()
                                   << " ports but its function has "
                                   << streamIdx << " stream parameters";
    if ((sawStream || sawIndex) && idxCount != static_cast<unsigned>(dims))
      return emitOpError("role '")
             << unit << "' has " << idxCount
             << " index parameters but the grid has " << dims << " dimensions";
  }
  if (ArrayAttr halo = getHaloAttr()) {
    for (Attribute haloAttr : halo) {
      auto task = llvm::dyn_cast<HaloAttr>(haloAttr);
      if (!task)
        continue; // payload kind is checked by verify()
      auto fn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
          *this, task.getUnit());
      if (!fn)
        return emitOpError("halo task references undefined function '")
               << task.getUnit().getValue() << "'";
      ArrayRef<Type> inputs = fn.getFunctionType().getInputs();
      // A loader reads a tensor row/column and streams it in: (memref, index,
      // stream). A drain consumes the exit edge: (stream).
      if (task.getKind() == "load") {
        int64_t operand = task.getOperand();
        if (inputs.size() != 3 || !llvm::isa<MemRefType>(inputs[0]) ||
            !llvm::isa<IndexType>(inputs[1]) ||
            !llvm::isa<allo::StreamType>(inputs[2]))
          return emitOpError("halo loader '")
                 << task.getUnit().getValue()
                 << "' must take (memref, index, stream)";
        if (operand >= 0 &&
            operand < static_cast<int64_t>(tensorTypes.size()) &&
            inputs[0] != tensorTypes[operand])
          return emitOpError("halo loader '")
                 << task.getUnit().getValue() << "' memref argument ("
                 << inputs[0] << ") does not match map tensor operand "
                 << operand << " (" << tensorTypes[operand] << ")";
      } else if (task.getKind() == "drain") {
        if (inputs.size() != 1 || !llvm::isa<allo::StreamType>(inputs[0]))
          return emitOpError("halo drain '")
                 << task.getUnit().getValue() << "' must take exactly (stream)";
      }
      // the halo stream (always the last parameter) must match the depth/type
      // of the boundary port it feeds/drains
      if (!inputs.empty())
        if (auto stream = llvm::dyn_cast<allo::StreamType>(inputs.back())) {
          auto abiIt = portAbi.find(task.getPort());
          if (abiIt != portAbi.end()) {
            if (static_cast<int64_t>(stream.getDepth()) != abiIt->second.first)
              return emitOpError("halo '")
                     << task.getUnit().getValue() << "' stream has depth "
                     << stream.getDepth() << " but port '" << task.getPort()
                     << "' declares depth " << abiIt->second.first;
            if (abiIt->second.second &&
                stream.getBaseType() != abiIt->second.second)
              return emitOpError("halo '")
                     << task.getUnit().getValue()
                     << "' stream element type does not match port '"
                     << task.getPort() << "'";
          }
        }
    }
  }
  return success();
}

} // namespace spmw
} // namespace mlir
