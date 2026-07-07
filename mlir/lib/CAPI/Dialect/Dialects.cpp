/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Dialect/Dialects.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/SPMW/SPMWDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Allo, allo, mlir::allo::AlloDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SPMW, spmw, mlir::spmw::SPMWDialect)