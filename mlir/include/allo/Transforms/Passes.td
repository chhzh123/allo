/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MLIR_PASSES
#define ALLO_MLIR_PASSES

include "mlir/Pass/PassBase.td"

def LoopTransformation : Pass<"loop-opt", "ModuleOp"> {
  let summary = "Loop transformation pass";
  let constructor = "mlir::allo::createLoopTransformationPass()";
}

def DataPlacement : Pass<"data-placement", "ModuleOp"> {
  let summary = "Data placement pass";
  let constructor = "mlir::allo::createDataPlacementPass()";
}

def AnyWidthInteger : Pass<"anywidth-integer", "ModuleOp"> {
  let summary = "Transform anywidth-integer input to 64-bit";
  let constructor = "mlir::allo::createAnyWidthIntegerPass()";
}

def MoveReturnToInput : Pass<"return-to-input", "ModuleOp"> {
  let summary = "Move return values to input argument list";
  let constructor = "mlir::allo::createMoveReturnToInputPass()";
}

def LegalizeCast : Pass<"legalize-cast", "ModuleOp"> {
  let summary = "Legalize cast operations";
  let constructor = "mlir::allo::createLegalizeCastPass()";
}

def RemoveStrideMap : Pass<"remove-stride-map", "ModuleOp"> {
  let summary = "Remove stride map from partitioned memref";
  let constructor = "mlir::allo::createRemoveStrideMapPass()";
}

def MemRefDCE : Pass<"memref-dce", "ModuleOp"> {
  let summary = "Remove MemRefs that are never loaded from";
  let constructor = "mlir::allo::createMemRefDCEPass()";
}

def CopyOnWrite : Pass<"copy-on-write", "ModuleOp"> {
  let summary = "Avoid copying data to new buffer until the old buffer have to be modified";
  let constructor = "mlir::allo::createCopyOnWritePass()";
}

def SPMWUnroll : Pass<"spmw-unroll", "ModuleOp"> {
  let summary = "Expand each spmw.map into one func.call per grid point";
  let constructor = "mlir::allo::createSPMWUnrollPass()";
}

def SPMWRolePartition : Pass<"spmw-role-partition", "ModuleOp"> {
  let summary = "Classify a spmw.map's grid points by role (attach spmw.partition counts)";
  let constructor = "mlir::allo::createSPMWRolePartitionPass()";
}

def SPMWResolveChannels : Pass<"spmw-resolve-channels", "ModuleOp"> {
  let summary = "Group a spmw.map's peer links into channel families (attach spmw.channel_families)";
  let constructor = "mlir::allo::createSPMWResolveChannelsPass()";
}

#endif // ALLO_MLIR_PASSES