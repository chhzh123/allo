/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SPMW_ATTRS
#define SPMW_ATTRS

include "allo/Dialect/SPMW/SPMWDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"

//===----------------------------------------------------------------------===//
// Interconnect links
//===----------------------------------------------------------------------===//

def SPMW_PeerLinkAttr : SPMW_Attr<"PeerLink", "peer_link"> {
  let summary = "A point-to-point interconnect link to a grid neighbor.";
  let description = [{
    A peer link binds a local `port` to the `peerPort` of the work unit reached by
    the affine `peerMap` applied to this unit's coordinate. `depth` is the FIFO
    depth; `elementType`, when present, is the element type carried by the channel.
  }];
  let parameters = (ins StringRefParameter<>:$port,
                        "::mlir::AffineMapAttr":$peerMap,
                        StringRefParameter<>:$peerPort,
                        "int64_t":$depth,
                        OptionalParameter<"::mlir::TypeAttr">:$elementType);
  let assemblyFormat = [{
    `<` `port` `=` $port `,` `map` `=` $peerMap `,` `peer` `=` $peerPort
    `,` `depth` `=` $depth (`,` `type` `=` $elementType^)? `>`
  }];
}

def SPMW_KeyLinkAttr : SPMW_Attr<"KeyLink", "key_link"> {
  let summary = "A rendezvous-by-key interconnect endpoint (a source or a sink).";
  let description = [{
    A key link binds a local `port` to a channel identified by `key`. `endpoint` is
    `"src"` or `"sink"`; endpoints sharing a key form one channel. `depth` is the
    FIFO depth; `elementType`, when present, is the element type carried.
  }];
  let parameters = (ins StringRefParameter<>:$port,
                        "::mlir::Attribute":$key,
                        StringRefParameter<>:$endpoint,
                        "int64_t":$depth,
                        OptionalParameter<"::mlir::TypeAttr">:$elementType);
  let assemblyFormat = [{
    `<` `port` `=` $port `,` `key` `=` $key `,` `end` `=` $endpoint
    `,` `depth` `=` $depth (`,` `type` `=` $elementType^)? `>`
  }];
}

//===----------------------------------------------------------------------===//
// Topology and roles
//===----------------------------------------------------------------------===//

def SPMW_TopologyAttr : SPMW_Attr<"Topology", "topology"> {
  let summary = "A grid plus the links that wire its work units.";
  let description = [{
    `grid` is the extent along each dimension, `dims` is the grid rank, and `links`
    is the array of `peer_link`/`key_link` attributes describing the interconnect.
  }];
  let parameters = (ins ArrayRefParameter<"int64_t">:$grid,
                        "int64_t":$dims,
                        "::mlir::ArrayAttr":$links);
  let assemblyFormat = [{
    `<` `grid` `=` `[` $grid `]` `,` `dims` `=` $dims `,` `links` `=` $links `>`
  }];
}

def SPMW_RoleAttr : SPMW_Attr<"Role", "role"> {
  let summary = "A boundary-specialized role and the missing links that select it.";
  let description = [{
    `unit` is the per-role function; `missing` is the set of boundary port names
    whose absence selects this role.
  }];
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$unit,
                        "::mlir::ArrayAttr":$missing);
  let assemblyFormat = "`<` `unit` `=` $unit `,` `missing` `=` $missing `>`";
}

#endif // SPMW_ATTRS
