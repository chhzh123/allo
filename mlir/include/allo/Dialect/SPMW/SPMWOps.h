/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_DIALECT_SPMW_SPMWOPS_H
#define ALLO_DIALECT_SPMW_SPMWOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "allo/Dialect/SPMW/SPMWAttrs.h"
#include "allo/Dialect/SPMW/SPMWDialect.h"

#define GET_OP_CLASSES
#include "allo/Dialect/SPMW/SPMWOps.h.inc"

#endif // ALLO_DIALECT_SPMW_SPMWOPS_H
