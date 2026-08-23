/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===- SPMWOps.cpp - spmw operations ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "allo/Dialect/SPMW/SPMWOps.h"
#include "allo/Dialect/AlloTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace mlir::spmw;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

/// Total number of grid points.
int64_t gridVolume(ArrayRef<int64_t> grid) {
  int64_t n = 1;
  for (int64_t extent : grid)
    n *= extent;
  return n;
}

/// The strings in an ArrayAttr, or failure if any element is not a string.
LogicalResult collectStrings(ArrayAttr arr, SmallVectorImpl<StringRef> &out) {
  for (Attribute a : arr) {
    auto s = llvm::dyn_cast<StringAttr>(a);
    if (!s)
      return failure();
    out.push_back(s.getValue());
  }
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

LogicalResult MapOp::verify() {
  TopologyAttr topo = getTopology();
  ArrayRef<int64_t> grid = topo.getGrid().asArrayRef();

  if (grid.empty())
    return emitOpError("grid must have at least one axis");
  for (int64_t extent : grid)
    if (extent <= 0)
      return emitOpError("grid extent must be positive, got ")
             << extent;

  // -- families ------------------------------------------------------------
  llvm::StringMap<FamilyAttr> byName;
  for (Attribute a : topo.getFamilies()) {
    auto fam = llvm::dyn_cast<FamilyAttr>(a);
    if (!fam)
      return emitOpError("topology families must be #spmw.family attributes");
    if (fam.getDepth() <= 0)
      return emitOpError("family '")
             << fam.getName() << "' has non-positive depth " << fam.getDepth();
    for (int64_t extent : fam.getShape().asArrayRef())
      if (extent <= 0)
        return emitOpError("family '")
               << fam.getName() << "' has a non-positive extent";
    if (!byName.insert({fam.getName(), fam}).second)
      return emitOpError("two families are named '") << fam.getName() << "'";
  }

  // -- port maps -----------------------------------------------------------
  // A port may appear more than once: routing is per site, so one port can
  // address different families at different sites.
  for (Attribute a : topo.getPortMaps()) {
    auto pm = llvm::dyn_cast<PortMapAttr>(a);
    if (!pm)
      return emitOpError("topology portMaps must be #spmw.port_map attributes");

    auto it = byName.find(pm.getFamily());
    if (it == byName.end())
      return emitOpError("port '")
             << pm.getPort() << "' addresses family '" << pm.getFamily()
             << "', which the topology does not declare";
    FamilyAttr fam = it->second;

    StringRef kind = pm.getKind();
    if (kind == "affine") {
      if (!pm.getOffset())
        return emitOpError("port '")
               << pm.getPort() << "' is affine but carries no offset";
      if (pm.getOffset().size() != static_cast<int64_t>(grid.size()))
        return emitOpError("port '")
               << pm.getPort() << "' has a rank-" << pm.getOffset().size()
               << " offset on a rank-" << grid.size() << " grid";
      if (fam.getShape().size() != (int64_t)grid.size())
        return emitOpError("family '")
               << fam.getName()
               << "' is addressed affinely, so its shape must match the grid's "
                  "rank";
    } else if (kind == "table") {
      if (!pm.getSlots())
        return emitOpError("port '")
               << pm.getPort() << "' is table-addressed but carries no slots";
      auto slots = llvm::dyn_cast<DenseIntElementsAttr>(pm.getSlots());
      if (!slots)
        return emitOpError("port '")
               << pm.getPort() << "' slots must be a dense integer table";
      if (slots.getNumElements() != gridVolume(grid))
        return emitOpError("port '")
               << pm.getPort() << "' has " << slots.getNumElements()
               << " slots for a grid of " << gridVolume(grid) << " points";
      // Every live slot must name a channel the family actually has. -1 marks a
      // site where the port is unbound, which is the honest encoding for a
      // boundary and must not be confused with channel zero.
      ArrayRef<int64_t> famShape = fam.getShape().asArrayRef();
      int64_t count = famShape.empty() ? 0 : famShape[0];
      for (const APInt &v : slots.getValues<APInt>()) {
        int64_t slot = v.getSExtValue();
        if (slot < -1 || slot >= count)
          return emitOpError("port '")
                 << pm.getPort() << "' names channel " << slot
                 << " in family '" << fam.getName() << "', which has " << count;
      }
    } else {
      return emitOpError("port '")
             << pm.getPort() << "' has addressing kind '" << kind
             << "'; expected \"affine\" or \"table\"";
    }
  }

  // -- roles ---------------------------------------------------------------
  if (getRoles().empty())
    return emitOpError("a map must declare at least one role");
  for (Attribute a : getRoles()) {
    auto role = llvm::dyn_cast<RoleAttr>(a);
    if (!role)
      return emitOpError("roles must be #spmw.role attributes");
    SmallVector<StringRef> names;
    if (failed(collectStrings(role.getPorts(), names)))
      return emitOpError("role '")
             << role.getUnit().getValue() << "' has a non-string port name";
    llvm::DenseSet<StringRef> seen;
    for (StringRef n : names)
      if (!seen.insert(n).second)
        return emitOpError("role '")
               << role.getUnit().getValue() << "' names port '" << n
               << "' twice";
    SmallVector<StringRef> missing;
    if (failed(collectStrings(role.getMissing(), missing)))
      return emitOpError("role '")
             << role.getUnit().getValue() << "' has a non-string missing entry";
  }

  // -- the class table -----------------------------------------------------
  auto classes = llvm::dyn_cast<DenseIntElementsAttr>(getClasses());
  if (!classes)
    return emitOpError("classes must be a dense integer table");
  if (classes.getNumElements() != gridVolume(grid))
    return emitOpError("classes has ")
           << classes.getNumElements() << " entries for a grid of "
           << gridVolume(grid) << " points";
  int64_t nRoles = static_cast<int64_t>(getRoles().size());
  for (const APInt &v : classes.getValues<APInt>()) {
    int64_t idx = v.getSExtValue();
    if (idx < 0 || idx >= nRoles)
      return emitOpError("classes names role ")
             << idx << ", but only " << nRoles << " are declared";
  }

  return success();
}

/// Check each role's function against the calling convention the map implies:
/// the tensors, then one `index` per grid axis, then one `!allo.stream` per
/// declared port in `ports` order, each matching its family's element type,
/// block shape and depth.
LogicalResult MapOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  TopologyAttr topo = getTopology();
  ArrayRef<int64_t> grid = topo.getGrid().asArrayRef();

  llvm::StringMap<FamilyAttr> famOfPort;
  llvm::StringMap<FamilyAttr> byName;
  for (Attribute a : topo.getFamilies()) {
    auto fam = llvm::cast<FamilyAttr>(a);
    byName.insert({fam.getName(), fam});
  }
  for (Attribute a : topo.getPortMaps()) {
    auto pm = llvm::cast<PortMapAttr>(a);
    auto it = byName.find(pm.getFamily());
    if (it != byName.end())
      famOfPort.insert({pm.getPort(), it->second});
  }

  for (Attribute a : getRoles()) {
    auto role = llvm::cast<RoleAttr>(a);
    auto fn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
        *this, role.getUnit());
    if (!fn)
      return emitOpError("role names '")
             << role.getUnit().getValue() << "', which is not a function here";

    SmallVector<StringRef> ports;
    (void)collectStrings(role.getPorts(), ports);

    ArrayRef<Type> inputs = fn.getFunctionType().getInputs();
    size_t nTensors = getTensors().size();
    size_t want = nTensors + grid.size() + ports.size();
    if (inputs.size() != want)
      return emitOpError("role '")
             << role.getUnit().getValue() << "' takes " << inputs.size()
             << " parameters; the map implies " << want << " ("
             << nTensors << " tensors, " << grid.size() << " pids, "
             << ports.size() << " streams)";

    for (size_t i = 0; i < grid.size(); ++i)
      if (!llvm::isa<IndexType>(inputs[nTensors + i]))
        return emitOpError("role '")
               << role.getUnit().getValue() << "' parameter "
               << (nTensors + i) << " should be the grid coordinate, an index";

    for (size_t i = 0; i < ports.size(); ++i) {
      Type t = inputs[nTensors + grid.size() + i];
      auto stream = llvm::dyn_cast<allo::StreamType>(t);
      if (!stream)
        return emitOpError("role '")
               << role.getUnit().getValue() << "' parameter for port '"
               << ports[i] << "' is not an !allo.stream";
      auto it = famOfPort.find(ports[i]);
      if (it == famOfPort.end())
        return emitOpError("role '")
               << role.getUnit().getValue() << "' declares port '" << ports[i]
               << "', which no port map mentions";
      FamilyAttr fam = it->second;
      if (stream.getDepth() != fam.getDepth())
        return emitOpError("port '")
               << ports[i] << "' is declared depth " << stream.getDepth()
               << " but its family '" << fam.getName() << "' is depth "
               << fam.getDepth();
      if (stream.getBaseType() != fam.getElementType().getValue())
        return emitOpError("port '")
               << ports[i] << "' carries a different type than its family '"
               << fam.getName() << "'";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Tablegen op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "allo/Dialect/SPMW/SPMWOps.cpp.inc"
