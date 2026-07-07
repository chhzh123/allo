/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SPMW_OPS
#define SPMW_OPS

include "allo/Dialect/SPMW/SPMWDialect.td"
include "allo/Dialect/SPMW/SPMWAttrs.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// spmw.rank
//===----------------------------------------------------------------------===//

def SPMW_RankOp : SPMW_Op<"rank", [Pure]> {
  let summary = "The compile-time grid coordinate of a work unit.";
  let description = [{
    Yields the coordinate of the enclosing work unit within its grid, one index
    result per grid dimension. This is the structural form of the frontend's
    `ctx.rank()`.
  }];
  let results = (outs Variadic<Index>:$ids);
  let assemblyFormat = "attr-dict `:` type($ids)";
}

//===----------------------------------------------------------------------===//
// spmw.map
//===----------------------------------------------------------------------===//

def SPMW_MapOp : SPMW_Op<"map", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Rolled spatial map of a work unit over a grid.";
  let description = [{
    Replicates a work unit over a grid, wired by a topology, without expanding to
    one function per grid point. It carries:
      * `tensors`  - the top-level memrefs the map reads and writes;
      * `topology` - the grid, its dimensionality, and the typed interconnect links;
      * `roles`    - an `spmw.role` per boundary specialization;
      * `fold`/`unroll` - optional time-multiplex / spatial-unroll factors.
  }];

  let arguments = (ins
    Variadic<AnyStaticShapeMemRef>:$tensors,
    SPMW_TopologyAttr:$topology,
    ArrayAttr:$roles,
    OptionalAttr<DenseI64ArrayAttr>:$fold,
    OptionalAttr<DenseI64ArrayAttr>:$unroll
  );

  let assemblyFormat = [{
    `(` $tensors `)`
    `topology` `=` $topology
    `roles` `=` $roles
    attr-dict
    `:` type($tensors)
  }];

  let hasVerifier = 1;
}

#endif // SPMW_OPS
