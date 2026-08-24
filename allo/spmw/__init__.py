# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW -- Single-Program, Multiple-Work-unit.

Write one work unit against a declared port contract, declare how the copies are
replicated and wired, and assemble hierarchy by placing components on fabrics::

    import allo.spmw as spmw

    class MacIO(spmw.Interface):
        west = spmw.In(float32);  north = spmw.In(float32)
        east = spmw.Out(float32); south = spmw.Out(float32)
        c    = spmw.MemOut(float32)

    @spmw.unit
    def pe(io: MacIO):
        acc: float32 = 0
        for k in range(K):
            a = io.west.get(); b = io.north.get()
            acc += a * b
            io.east.put(a); io.south.put(b)
        io.c = acc

    @spmw.fabric
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        P = spmw.place(pe, on=spmw.mesh(MacIO, (M, N)))
        spmw.stream_in(A, into=P.west, index=(P.rows, ...))
        spmw.stream_in(B, into=P.north, index=(..., P.cols))
        spmw.gather(C, from_=P.c)

The elaboration core imports nothing from Allo proper, so contracts, topologies
and bindings can be checked without a built compiler; ``build`` pulls in the
backend on first use.
"""

from .bindings import (
    copy,
    gather,
    link,
    phase,
    scatter,
    shard,
    stationary,
    stream_in,
)
from .bricks import (
    FIFO,
    RAM,
    Brick,
    Layout,
    Tensor,
    banked,
    mem,
    replicate,
    shared,
    xor_bank,
)
from .component import Fabric, Site, Unit, fabric, unit
from .errors import (
    SPMWBindingError,
    SPMWDirectionError,
    SPMWError,
    SPMWMemoryError,
    SPMWOwnershipError,
    SPMWPlacementError,
    SPMWTopologyError,
    SPMWTypeError,
    SPMWUnboundError,
)
from .graph import Elaborated, elaborate
from .index import Axis, split
from .iface import Interface, interface, structural
from .placement import Bundle, MemGrid, Placement, place
from .ports import In, Mem, MemIn, MemOut, Out
from .topology import Grid, Topology, key, mesh, ring, to


# The implementation lives in `driver`, not `build`: a submodule named `build`
# would be bound as an attribute of this package on first import and shadow the
# function below, so the first call would work and every one after it would find
# the module instead.
def build(fabric_fn, target="simulator", **kwargs):
    """Elaborate a fabric and compile it for ``target``.

    Imported lazily so that ``import allo.spmw`` costs nothing but the frontend.
    """
    from .driver import build as _build

    return _build(fabric_fn, target=target, **kwargs)


def source(fabric_fn, **kwargs):
    """The dataflow program a fabric lowers to, without compiling it.

    Useful for asserting structural properties of the emitted program -- that
    its body count tracks the role count rather than the grid, for instance.
    """
    from .driver import source as _source

    return _source(fabric_fn, **kwargs)


def customize(fabric_fn, **kwargs):
    """Elaborate a fabric and return the schedule, before any backend runs."""
    from .driver import (
        customize as _customize,
    )

    return _customize(fabric_fn, **kwargs)


__all__ = [
    # contracts
    "Interface",
    "interface",
    "structural",
    "In",
    "Out",
    "MemIn",
    "MemOut",
    "Mem",
    # components
    "unit",
    "fabric",
    "Unit",
    "Fabric",
    "Site",
    # topologies
    "Topology",
    "Grid",
    "mesh",
    "ring",
    "to",
    "key",
    # placement
    "place",
    "Placement",
    "Bundle",
    "MemGrid",
    "split",
    "Axis",
    # memory
    "mem",
    "RAM",
    "FIFO",
    "Brick",
    "Tensor",
    "Layout",
    "banked",
    "replicate",
    "shared",
    "xor_bank",
    # bindings
    "stream_in",
    "gather",
    "scatter",
    "shard",
    "stationary",
    "copy",
    "link",
    "phase",
    # compilation
    "build",
    "customize",
    "source",
    "elaborate",
    "Elaborated",
    # diagnostics
    "SPMWError",
    "SPMWDirectionError",
    "SPMWTypeError",
    "SPMWOwnershipError",
    "SPMWTopologyError",
    "SPMWPlacementError",
    "SPMWBindingError",
    "SPMWMemoryError",
    "SPMWUnboundError",
]
