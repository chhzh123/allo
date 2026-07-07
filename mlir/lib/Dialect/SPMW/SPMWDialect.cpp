/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===- SPMWDialect.cpp - spmw dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "allo/Dialect/SPMW/SPMWDialect.h"
#include "allo/Dialect/SPMW/SPMWAttrs.h"
#include "allo/Dialect/SPMW/SPMWOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::spmw;

#include "allo/Dialect/SPMW/SPMWDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Tablegen attribute definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "allo/Dialect/SPMW/SPMWAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
void SPMWDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "allo/Dialect/SPMW/SPMWOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "allo/Dialect/SPMW/SPMWAttrs.cpp.inc"
      >();
}
