# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_public_c_api_library(MLIRAlloCAPI
  Dialects.cpp
  AlloTypes.cpp
  AlloAttributes.cpp
  Registration.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/Passes.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/LoopTransformations.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCAPIIR
  MLIRSupport
  MLIRAllo
  MLIRAlloPasses
  MLIRAlloConversion
  )
