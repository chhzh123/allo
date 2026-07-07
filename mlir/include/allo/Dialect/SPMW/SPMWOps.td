/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SPMW_OPS
#define SPMW_OPS

include "allo/Dialect/SPMW/SPMWDialect.td"
include "mlir/IR/BuiltinAttributes.td"
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

def SPMW_MapOp : SPMW_Op<"map"> {
  let summary = "Rolled spatial map of a work unit over a grid.";
  let description = [{
    Replicates a work unit over a grid, wired by a topology, without expanding to
    one function per grid point. It carries:
      * `tensors`   - the top-level memrefs the map reads and writes;
      * `grid`      - the extent along each grid dimension;
      * `topologyDims` - the dimensionality of the topology (must equal the grid
        rank);
      * `links`     - one affine map per interconnect link (the peer/key map);
      * `roles`     - a symbol reference to the per-role function for each role;
      * `fold`/`unroll` - optional time-multiplex / spatial-unroll factors.
  }];

  let arguments = (ins
    Variadic<AnyStaticShapeMemRef>:$tensors,
    DenseI64ArrayAttr:$grid,
    I64Attr:$topologyDims,
    ArrayAttr:$links,
    ArrayAttr:$roles,
    OptionalAttr<DenseI64ArrayAttr>:$fold,
    OptionalAttr<DenseI64ArrayAttr>:$unroll
  );

  let assemblyFormat = [{
    `(` $tensors `)`
    `grid` `=` $grid
    `dims` `=` $topologyDims
    `links` `=` $links
    `roles` `=` $roles
    attr-dict
    `:` type($tensors)
  }];

  let hasVerifier = 1;
}

#endif // SPMW_OPS
