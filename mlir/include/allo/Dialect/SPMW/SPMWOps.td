/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SPMW_OPS
#define SPMW_OPS

include "allo/Dialect/SPMW/SPMWAttrs.td"
include "allo/Dialect/SPMW/SPMWDialect.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SPMW_MapOp : SPMW_Op<"map",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "A work unit mapped over a grid, rolled.";
  let description = [{
    Carries the whole structure of a spatial design -- the grid, the channel
    families, how each port addresses them, and which body runs where -- without
    expanding it. One body per role reaches code generation, not one per grid
    point, so the tool schedules a number of functions bounded by the role count
    rather than by the array's size.

    `classes` is the grid-shaped table naming which entry of `roles` runs at each
    site. It is materialised rather than recomputed because the emitter needs it
    per site and rediscovering it would mean re-running the whole link analysis.

    Example: a 2-D mesh has nine site signatures at any size, so `roles` has nine
    entries whether the grid is 4x4 or 64x64.
  }];

  let arguments = (ins
    Variadic<AnyStaticShapeMemRef>:$tensors,
    SPMW_TopologyAttr:$topology,
    ArrayAttr:$roles,
    ElementsAttr:$classes
  );

  let assemblyFormat = [{
    `(` $tensors `)` `topology` `=` $topology `roles` `=` $roles
    `classes` `=` $classes attr-dict (`:` type($tensors)^)?
  }];

  let hasVerifier = 1;
}

#endif // SPMW_OPS
