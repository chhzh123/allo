/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
namespace {
#define GEN_PASS_REGISTRATION
#include "allo/Conversion/Passes.h.inc"
} // end namespace

void mlir::allo::registerAlloConversionPasses() { ::registerPasses(); }
