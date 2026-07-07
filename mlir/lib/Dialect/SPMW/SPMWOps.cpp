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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::spmw;

#define GET_OP_CLASSES
#include "allo/Dialect/SPMW/SPMWOps.cpp.inc"

namespace mlir {
namespace spmw {

LogicalResult MapOp::verify() {
  int64_t gridRank = static_cast<int64_t>(getGrid().size());
  int64_t topologyDims = static_cast<int64_t>(getTopologyDims());
  if (gridRank != topologyDims)
    return emitOpError("grid rank (")
           << gridRank << ") does not match topology dims (" << topologyDims
           << ")";
  for (Attribute role : getRoles())
    if (!llvm::isa<FlatSymbolRefAttr>(role))
      return emitOpError("each role must be a flat symbol reference");
  for (Attribute link : getLinks())
    if (!llvm::isa<AffineMapAttr>(link))
      return emitOpError("each link must be an affine map");
  return success();
}

} // namespace spmw
} // namespace mlir
