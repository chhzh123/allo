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

def SPMW_EdgeLinkAttr : SPMW_Attr<"EdgeLink", "edge_link"> {
  let summary = "An explicit per-coordinate interconnect edge for an irregular topology.";
  let description = [{
    An edge link binds the local `port` at grid coordinate `at` to the `peerPort` of
    the work unit at coordinate `peer` (which may be out of bounds, marking a boundary
    edge). Unlike `peer_link`, each concrete edge is listed explicitly, so an irregular
    topology whose reciprocal peer port varies by coordinate -- e.g. a binary reduction
    tree's `up` edge, which peers a parent's `left` or `right` by child parity -- is
    represented faithfully. `depth` is the FIFO depth; `elementType`, when present, is
    the element type carried.
  }];
  let parameters = (ins StringRefParameter<>:$port,
                        ArrayRefParameter<"int64_t">:$at,
                        ArrayRefParameter<"int64_t">:$peer,
                        StringRefParameter<>:$peerPort,
                        "int64_t":$depth,
                        OptionalParameter<"::mlir::TypeAttr">:$elementType);
  let assemblyFormat = [{
    `<` `port` `=` $port `,` `at` `=` `[` $at `]` `,` `peer` `=` `[` $peer `]`
    `,` `peer_port` `=` $peerPort `,` `depth` `=` $depth
    (`,` `type` `=` $elementType^)? `>`
  }];
}

//===----------------------------------------------------------------------===//
// Topology and roles
//===----------------------------------------------------------------------===//

def SPMW_TopologyAttr : SPMW_Attr<"Topology", "topology"> {
  let summary = "A grid plus the links that wire its work units.";
  let description = [{
    `grid` is the extent along each dimension, `dims` is the grid rank, and `links`
    is the array of `peer_link` / `key_link` / `edge_link` attributes describing the
    interconnect (affine grid-neighbor links, rendezvous-by-key endpoints, and
    explicit per-coordinate edges of an irregular topology, respectively).
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
    whose absence selects this role. `ports`, when present, names the local port
    each of the role func's stream parameters carries, in signature order -- the
    declaration-derived stream ABI the unroll pass binds channels by. `predicate`,
    when present, is an affine map from the grid coordinate to an integer tag: grid
    points that share a link-presence class but differ in this tag are distinct role
    instances (e.g. a checkerboard interior split by `(i, j) -> (i + j) mod 2`).
  }];
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$unit,
                        "::mlir::ArrayAttr":$missing,
                        OptionalParameter<"::mlir::ArrayAttr">:$ports,
                        OptionalParameter<"::mlir::AffineMapAttr">:$predicate);
  let assemblyFormat = [{
    `<` `unit` `=` $unit `,` `missing` `=` $missing
    (`,` `ports` `=` $ports^)? (`,` `predicate` `=` $predicate^)? `>`
  }];
}

def SPMW_HaloAttr : SPMW_Attr<"Halo", "halo"> {
  let summary = "A boundary loader/drain task attached to a grid edge.";
  let description = [{
    `unit` is the loader/drain function. The task applies at each grid point where
    `port` is missing (an off-grid boundary port), wiring `unit` to the same channel
    that point's compute role uses on `port`. `kind` is `"load"` (a producer that
    streams `operand` into the array) or `"drain"` (a consumer at the exit edge).
    `axis` is the grid dimension whose coordinate feeds a loader's position argument;
    `operand` indexes the map tensors a loader reads.
  }];
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$unit,
                        StringRefParameter<>:$port,
                        StringRefParameter<>:$kind,
                        "int64_t":$axis,
                        "int64_t":$operand);
  let assemblyFormat = [{
    `<` `unit` `=` $unit `,` `port` `=` $port `,` `kind` `=` $kind
    `,` `axis` `=` $axis `,` `operand` `=` $operand `>`
  }];
}

def SPMW_ChannelAttr : SPMW_Attr<"Channel", "channel"> {
  let summary = "A named inter-unit FIFO channel connecting two mapped units.";
  let description = [{
    `name` identifies the channel; `elementType` is the payload type it carries and
    `depth` its FIFO depth. `producer` and `consumer` are the role funcs of the two
    maps that put to and get from it -- so a later pass can resolve the channel
    without re-scanning bodies. Unlike a topology link (which wires grid neighbors of
    one map), a channel connects two distinct maps, so a rolled pipeline region carries
    its channels alongside the maps.
  }];
  let parameters = (ins StringRefParameter<>:$name,
                        "::mlir::TypeAttr":$elementType,
                        "int64_t":$depth,
                        "::mlir::FlatSymbolRefAttr":$producer,
                        "::mlir::FlatSymbolRefAttr":$consumer);
  let assemblyFormat = [{
    `<` `name` `=` $name `,` `type` `=` $elementType `,` `depth` `=` $depth
    `,` `producer` `=` $producer `,` `consumer` `=` $consumer `>`
  }];
}

def SPMW_MemoryAttr : SPMW_Attr<"Memory", "memory"> {
  let summary = "A tensor operand pinned to a logical memory resource.";
  let description = [{
    `tensor` is the operand name; `resource` is the concrete memory resource the
    logical `space=` level resolved to (e.g. "URAM", "BRAM", "AUTO"). `bankAxis`,
    when non-negative, is the tensor axis the buffer is banked along. This carries
    the resolved logical-memory model to the target so it can honor `resource=`.
  }];
  let parameters = (ins StringRefParameter<>:$tensor,
                        StringRefParameter<>:$resource,
                        "int64_t":$bankAxis);
  let assemblyFormat = [{
    `<` `tensor` `=` $tensor `,` `resource` `=` $resource
    `,` `bank_axis` `=` $bankAxis `>`
  }];
}

#endif // SPMW_ATTRS
