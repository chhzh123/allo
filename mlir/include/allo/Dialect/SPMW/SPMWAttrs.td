/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SPMW_ATTRS
#define SPMW_ATTRS

include "allo/Dialect/SPMW/SPMWDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// Channel families
//===----------------------------------------------------------------------===//

def SPMW_FamilyAttr : SPMW_Attr<"Family", "family"> {
  let summary = "One declarable array of streams.";
  let description = [{
    A family is what a set of channels becomes once it is declared: an array of
    streams with a shape, an element type and a depth. `block` is the shape of
    one *token* -- empty for a scalar channel, `[2]` for a channel carrying a
    complex sample -- which is separate from `shape`, the extent of the array
    itself.
  }];
  let parameters = (ins StringRefParameter<>:$name,
                        "::mlir::TypeAttr":$elementType,
                        "::mlir::DenseI64ArrayAttr":$block,
                        "int64_t":$depth,
                        "::mlir::DenseI64ArrayAttr":$shape);
  let assemblyFormat = [{
    `<` `name` `=` $name `,` `type` `=` $elementType `,` `block` `=` $block
    `,` `depth` `=` $depth `,` `shape` `=` $shape `>`
  }];
}

//===----------------------------------------------------------------------===//
// How a port reaches its family
//===----------------------------------------------------------------------===//

def SPMW_PortMapAttr : SPMW_Attr<"PortMap", "port_map"> {
  let summary = "Which family a port addresses, and how it computes the index.";
  let description = [{
    Two spellings, because the flagship shapes are affine and should stay
    readable. `kind = "affine"` means every site addresses `coord + offset`, and
    `offset` is that constant displacement -- a mesh's east port is `[0, 1]`.
    `kind = "table"` means the mapping has no closed form -- a permutation, a
    wrap, a broadcast -- and `slots` gives the channel index per grid point,
    with -1 where the port is unbound.

    A port may appear more than once with different families: routing is a
    per-*site* question, and a site's arm is selected on both its signature and
    its routing.
  }];
  let parameters = (ins StringRefParameter<>:$port,
                        StringRefParameter<>:$family,
                        StringRefParameter<>:$kind,
                        OptionalParameter<"::mlir::DenseI64ArrayAttr">:$offset,
                        OptionalParameter<"::mlir::ElementsAttr">:$slots);
  let assemblyFormat = [{
    `<` `port` `=` $port `,` `family` `=` $family `,` `kind` `=` $kind
    (`,` `offset` `=` $offset^)? (`,` `slots` `=` $slots^)? `>`
  }];
}

//===----------------------------------------------------------------------===//
// Topology and roles
//===----------------------------------------------------------------------===//

def SPMW_TopologyAttr : SPMW_Attr<"Topology", "topology"> {
  let summary = "A grid, the families over it, and how ports address them.";
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$grid,
                        "::mlir::ArrayAttr":$families,
                        "::mlir::ArrayAttr":$portMaps);
  let assemblyFormat = [{
    `<` `grid` `=` $grid `,` `families` `=` $families
    `,` `ports` `=` $portMaps `>`
  }];
}

def SPMW_RoleAttr : SPMW_Attr<"Role", "role"> {
  let summary = "One body, and the calling convention its sites use.";
  let description = [{
    `unit` names a `func.func` sibling rather than a region, so the existing HLS
    emitter already turns it into a C++ function and only the instantiation is
    new work. `ports` fixes the order the stream parameters appear in, and
    `missing` records the ports this body's sites do not have -- the site
    signature, which is what selects the body in the first place.
  }];
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$unit,
                        "::mlir::ArrayAttr":$missing,
                        "::mlir::ArrayAttr":$ports);
  let assemblyFormat = [{
    `<` `unit` `=` $unit `,` `missing` `=` $missing `,` `ports` `=` $ports `>`
  }];
}

#endif // SPMW_ATTRS
