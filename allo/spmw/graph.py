# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fabric elaboration -- running a declarative body once to build a graph.

A ``for`` in a fabric is structure *generation*; a ``for`` in a unit is
*iteration*.  This module runs the former: it executes the fabric body with a
collector active, gathering placements, bricks and bindings, then checks the
result against the rules the design states.
"""

import inspect

from .bricks import Brick, Tensor
from .component import Fabric
from .errors import (
    SPMWBindingError,
    SPMWMemoryError,
    SPMWUnboundError,
)
from .index import IndexMap, check_bounds
from .placement import Bundle, MemGrid
from .ports import STREAM, IN, READ

_STACK = []


def current_fabric(required=True):
    """The fabric currently elaborating, if any."""
    if _STACK:
        return _STACK[-1]
    if required:
        raise SPMWBindingError(
            "this verb is only meaningful inside an @spmw.fabric body, which is "
            "where structure is declared."
        )
    return None


class Binding:
    """One declared movement or aliasing between a tensor/brick and ports."""

    __slots__ = ("kind", "source", "target", "imap", "phase", "extras")

    def __init__(self, kind, source, target, imap=None, phase=None, **extras):
        self.kind = kind
        self.source = source
        self.target = target
        self.imap = imap
        self.phase = phase
        self.extras = extras

    def __repr__(self):
        return f"<{self.kind} {self.source} -> {self.target}>"


class Phase:
    """A fill-then-use epoch.  Orders a fabric's bindings; lowers to a barrier."""

    __slots__ = ("name", "fabric")

    def __init__(self, name):
        self.name = name
        self.fabric = None

    def __enter__(self):
        self.fabric = current_fabric()
        self.fabric.phases.append(self.name)
        self.fabric.phase_stack.append(self.name)
        return self

    def __exit__(self, *exc):
        self.fabric.phase_stack.pop()
        return False


class Elaborated:
    """The graph a fabric body produced."""

    def __init__(self, fabric, tensors):
        self.fabric = fabric
        self.tensors = tensors
        self.placements = []
        self.bricks = []
        self.bindings = []
        self.phases = []
        self.phase_stack = []

    @property
    def phase(self):
        return self.phase_stack[-1] if self.phase_stack else None

    def record(self, binding):
        binding.phase = self.phase
        self.bindings.append(binding)
        return binding

    # -- queries used by the checks and the lowering -----------------------

    def bindings_for(self, placement, port):
        """Every binding touching one placement's port."""
        out = []
        for b in self.bindings:
            for side in (b.source, b.target):
                if (
                    isinstance(side, (Bundle, MemGrid))
                    and side.placement is placement
                    and side.port is port
                ):
                    out.append(b)
                    break
        return out

    def __repr__(self):
        return (
            f"<elaborated {self.fabric.name}: {len(self.placements)} placements, "
            f"{len(self.bindings)} bindings, {len(self.bricks)} bricks>"
        )


def elaborate(fabric, tensor_specs=None):
    """Run a fabric body once and return its graph."""
    if not isinstance(fabric, Fabric):
        raise SPMWBindingError(
            f"elaborate expects an @spmw.fabric, got {type(fabric).__name__}."
        )
    tensors = _tensor_args(fabric, tensor_specs)
    graph = Elaborated(fabric, tensors)
    _STACK.append(graph)
    try:
        fabric.fn(*tensors.values())
    finally:
        _STACK.pop()
    check(graph)
    return graph


def _tensor_args(fabric, tensor_specs):
    """Top-level fabric arguments are bricks at the host boundary."""
    tensors = {}
    params = list(fabric.signature.parameters.values())
    for pos, param in enumerate(params):
        ann = param.annotation
        if tensor_specs and param.name in tensor_specs:
            dtype, shape = tensor_specs[param.name]
        elif ann is inspect.Parameter.empty:
            raise SPMWBindingError(
                f"fabric `{fabric.name}`'s argument `{param.name}` has no type "
                f"annotation; a top-level tensor's shape is part of the host "
                f"contract."
            )
        else:
            dtype = getattr(ann, "dtype", ann)
            shape = tuple(getattr(ann, "shape", ()))
        tensors[param.name] = Tensor(param.name, dtype, shape, pos)
    return tensors


def register_placement(placement):
    fab = current_fabric(required=False)
    if fab is not None and placement not in fab.placements:
        fab.placements.append(placement)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check(graph):
    """Everything the design says must hold once a fabric has elaborated."""
    _check_unbound_ins(graph)
    _check_memories_bound(graph)
    _check_phase_writers(graph)
    _check_writes_tile(graph)


def _covered_ports(graph, placement):
    """Ports of a placement that some binding feeds or drains."""
    covered = set()
    for b in graph.bindings:
        for side in (b.source, b.target):
            if isinstance(side, (Bundle, MemGrid)) and side.placement is placement:
                covered.add(side.port)
    return covered


def _check_unbound_ins(graph):
    """Every unbound ``In`` must be covered by a binding or a role.

    Reading from nowhere is undefined; writing to nowhere is a no-op.  That
    asymmetry is why only ``In`` forces this.
    """
    for placement in graph.placements:
        covered = _covered_ports(graph, placement)
        for (
            port,
            bundle,
        ) in placement._bundles.items():  # pylint: disable=protected-access
            if port.direction != IN or port.protocol != STREAM:
                continue
            if port in covered:
                continue
            # A site whose assigned body never touches the port is fine.
            readers = [
                site for site in bundle.sites if port in _body_reads(placement, site)
            ]
            if readers:
                raise SPMWUnboundError(
                    f"`{placement.name}.{port.name}` is unbound at "
                    f"{len(bundle.sites)} site(s) -- e.g. {readers[0]} -- and no "
                    f"binding feeds it. Cover it with spmw.stream_in / spmw.link / "
                    f"spmw.scatter, or give those sites a role declaring "
                    f"unbound=({port.name},)."
                )


def _body_reads(placement, site):
    body = placement.roles.get(site)
    if body is None:
        return frozenset()
    unit = placement.component
    return unit.reads(body)


def _check_memories_bound(graph):
    """Every ``MemIn`` bound; an unbound ``MemOut`` is a warning, not an error."""
    for placement in graph.placements:
        covered = _covered_ports(graph, placement)
        for (
            port,
            _grid,
        ) in placement._memgrids.items():  # pylint: disable=protected-access
            if port.access == READ and port not in covered:
                raise SPMWMemoryError(
                    f"`{placement.name}.{port.name}` is {port.type_str()} but "
                    f"nothing supplies it. Bind it with spmw.shard or "
                    f"spmw.stationary."
                )


def _check_phase_writers(graph):
    """Within a phase, a shared brick's clients are all readers or one writer."""
    per_phase = {}
    for b in graph.bindings:
        target = b.target
        brick = target if isinstance(target, Brick) else None
        if brick is None:
            continue
        per_phase.setdefault((b.phase, brick), []).append(b)
    for (phase, brick), writers in per_phase.items():
        if len(writers) > 1:
            where = f"phase {phase!r}" if phase else "the fabric"
            raise SPMWMemoryError(
                f"`{brick.name}` has {len(writers)} writers in {where}. Within a "
                f"phase a shared brick's clients must be all readers, or a single "
                f"writer with no readers. Separate them with spmw.phase(), or bank "
                f"the brick so each writer owns one."
            )


def bind_check(imap, bundle, tensor, where, extent=None, write=False, block=()):
    """Bounds-check a binding's index map over its whole domain."""
    members = [
        (site, dict(bundle.placement.env(site), __coords__=site))
        for site in bundle.sites
    ]
    return check_bounds(
        imap, members, tensor.shape, extent, where, write=write, block=block
    )


def make_map(index, rank, where):
    """Normalise an ``index=`` argument into an :class:`IndexMap`."""
    if index is None:
        return None
    try:
        return IndexMap(index, rank)
    except SPMWBindingError as err:
        raise SPMWBindingError(f"{where}: {err}") from err


__all__ = [
    "Binding",
    "Elaborated",
    "Phase",
    "bind_check",
    "check",
    "current_fabric",
    "elaborate",
    "make_map",
    "register_placement",
]


def _check_writes_tile(graph):
    """Across every writer of a tensor, the union must tile it exactly.

    Several writers may target one tensor -- two placements, or one placement
    across phases -- so exhaustiveness is a property of the union, not of any
    single binding. Each map was already checked against its own slice; what is
    left is that the slices are pairwise disjoint and jointly exhaustive.
    """
    per_tensor = {}
    for b in graph.bindings:
        written = b.extras.get("writes")
        if written is None:
            continue
        tensor = b.target
        seen, block = written
        per_tensor.setdefault(tensor, []).append((b, seen, block))

    for tensor, entries in per_tensor.items():
        total = 1
        for bound in tensor.shape:
            total *= bound
        owner = {}
        count = 0
        for binding, seen, block in entries:
            per_site = 1
            for bound in block:
                per_site *= bound
            for idx in seen:
                if idx in owner and owner[idx] is not binding:
                    raise SPMWBindingError(
                        f"`{tensor.name}`{list(idx)} is written by two bindings: "
                        f"{owner[idx].kind} and {binding.kind}. Writers into one "
                        f"tensor must bind pairwise-disjoint slices."
                    )
                owner[idx] = binding
            count += len(seen) * per_site
        if count != total:
            missing = _first_gap(owner, tensor.shape)
            raise SPMWBindingError(
                f"`{tensor.name}` is {list(tensor.shape)} but its writers cover "
                f"{count} of {total} elements. {missing} Every output element "
                f"must have exactly one writer -- supply the rest from another "
                f"phase, a wider instantiation, or a second placement owning the "
                f"other slice."
            )


def _first_gap(owner, shape):
    """Name the uncovered range on the first axis that has one."""
    for axis, bound in enumerate(shape):
        hit = {idx[axis] for idx in owner if axis < len(idx)}
        if not hit:
            continue
        gap = [v for v in range(bound) if v not in hit]
        if gap:
            return f"Axis {axis} is uncovered at [{gap[0]}, {gap[-1] + 1})."
    return ""
