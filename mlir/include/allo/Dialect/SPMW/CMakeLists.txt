# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS SPMWOps.td)
mlir_tablegen(SPMWOps.h.inc -gen-op-decls)
mlir_tablegen(SPMWOps.cpp.inc -gen-op-defs)
mlir_tablegen(SPMWDialect.h.inc -gen-dialect-decls -dialect=spmw)
mlir_tablegen(SPMWDialect.cpp.inc -gen-dialect-defs -dialect=spmw)
add_public_tablegen_target(MLIRSPMWOpsIncGen)

set(LLVM_TARGET_DEFINITIONS SPMWAttrs.td)
mlir_tablegen(SPMWAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=spmw)
mlir_tablegen(SPMWAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=spmw)
add_public_tablegen_target(MLIRSPMWAttrsIncGen)

# used to generate all the above targets at once
add_custom_target(MLIRSPMWIncGen
        DEPENDS
        MLIRSPMWOpsIncGen
        MLIRSPMWAttrsIncGen
)

add_mlir_doc(SPMWDialect SPMWDialect SPMW/ -gen-dialect-doc)
add_mlir_doc(SPMWOps SPMWOps SPMW/ -gen-op-doc)
