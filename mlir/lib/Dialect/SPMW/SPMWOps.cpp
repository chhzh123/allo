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
#include "allo/Dialect/SPMW/SPMWAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::spmw;

#define GET_OP_CLASSES
#include "allo/Dialect/SPMW/SPMWOps.cpp.inc"

namespace mlir {
namespace spmw {

LogicalResult MapOp::verify() {
  TopologyAttr topology = getTopology();
  int64_t dims = topology.getDims();
  ArrayRef<int64_t> grid = topology.getGrid();
  if (static_cast<int64_t>(grid.size()) != dims)
    return emitOpError("grid rank (")
           << grid.size() << ") does not match topology dims (" << dims << ")";

  // Each channel key must have exactly one source and one sink.
  llvm::DenseMap<Attribute, std::pair<int, int>> keyCounts;
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
    } else {
      return emitOpError(
          "topology link must be a peer_link or key_link attribute");
    }
  }
  for (const auto &entry : keyCounts) {
    int numSrc = entry.second.first;
    int numSink = entry.second.second;
    if (numSrc != 1 || numSink != 1)
      return emitOpError(
                 "channel key must have exactly one src and one sink; got ")
             << numSrc << " src and " << numSink << " sink";
  }

  for (Attribute roleAttr : getRoles())
    if (!llvm::isa<RoleAttr>(roleAttr))
      return emitOpError("each role must be an spmw.role attribute");

  if (ArrayAttr halo = getHaloAttr()) {
    int64_t numTensors = static_cast<int64_t>(getTensors().size());
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
  for (Attribute roleAttr : getRoles()) {
    auto role = llvm::dyn_cast<RoleAttr>(roleAttr);
    if (!role)
      continue; // payload kind is checked by verify()
    auto fn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this,
                                                                role.getUnit());
    if (!fn)
      return emitOpError("role references undefined function '")
             << role.getUnit().getValue() << "'";
    // A role func names the map's tensors first (memref args), then its
    // per-instantiation parameters (PID indices, streams). Its leading memref
    // args must therefore be a prefix of the map's tensor operands, matching
    // type by type.
    unsigned ti = 0;
    bool sawNonMemref = false;
    for (Type input : fn.getFunctionType().getInputs()) {
      if (!llvm::isa<MemRefType>(input)) {
        // the per-instantiation parameters (PID indices, streams) follow the
        // tensors
        sawNonMemref = true;
        continue;
      }
      if (sawNonMemref)
        return emitOpError("role '")
               << role.getUnit().getValue()
               << "' has a memref argument after a non-memref parameter; a "
                  "role's map tensors must precede its per-instantiation "
                  "parameters";
      if (ti >= tensorTypes.size())
        return emitOpError("role '") << role.getUnit().getValue()
                                     << "' takes more memref arguments than "
                                        "the map has tensor operands";
      if (input != tensorTypes[ti])
        return emitOpError("role '")
               << role.getUnit().getValue() << "' memref argument " << ti
               << " (" << input << ") does not match map tensor operand " << ti
               << " (" << tensorTypes[ti] << ")";
      ++ti;
    }
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
    }
  }
  return success();
}

} // namespace spmw
} // namespace mlir
