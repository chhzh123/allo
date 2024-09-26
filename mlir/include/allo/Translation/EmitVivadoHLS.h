/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_EMITVIVADOHLS_H
#define ALLO_TRANSLATION_EMITVIVADOHLS_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace allo {

LogicalResult emitVivadoHLS(ModuleOp module, llvm::raw_ostream &os);
void registerEmitVivadoHLSTranslation();

} // namespace allo
} // namespace mlir

#endif // ALLO_TRANSLATION_EMITVIVADOHLS_H