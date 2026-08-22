# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fabric elaboration -- running a declarative body once to build a graph.

A ``for`` in a fabric is structure *generation*; a ``for`` in a unit is
*iteration*.  This module runs the former: it executes the fabric body with a
collector active, gathering placements, bricks and bindings, then checks the
result against the rules the design states.
"""

import inspect

from .bricks import Brick, Tensor, TensorView
from .component import Fabric
from .errors import (
    SPMWBindingError,
    SPMWMemoryError,
    SPMWUnboundError,
)
from .index import Expr, IndexMap, check_bounds
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
        expand(graph)
    finally:
        _STACK.pop()
    check(graph)
    return graph


def expand(graph):
    """Replace each placed fabric by running its body once per site.

    A fabric placed on a topology is a composite component: what it exposes is
    the same kind of thing a unit declares, so expanding it is just running its
    declarative body with ``io`` bound to the slice this site owns. The
    placements and bindings it makes land in this same graph, which is what makes
    hierarchy need no new mechanism.
    """
    pending = [
        p
        for p in graph.placements
        if isinstance(p.component, Fabric) and not p.expanded
    ]
    for placement in pending:
        placement.expanded = True
        for site in placement.sites():
            io = _SubIO(_site_views(graph, placement, site))
            placement.component.fn(io)
    if pending:
        # A sub-fabric may itself place one, so keep going until none are left.
        expand(graph)


def _site_views(graph, placement, site):
    """The slice of each parent tensor that one site of a placed fabric owns."""
    views = {}
    for port in placement.iface.ports():
        binding = _binding_for(graph, placement, port)
        if binding is None:
            raise SPMWBindingError(
                f"`{placement.name}.{port.name}` is a port of a placed fabric but "
                f"nothing binds it, so the site has no data to work on."
            )
        source = (
            binding.source
            if isinstance(binding.target, (Bundle, MemGrid))
            else binding.target
        )
        views[port.name] = _slice_of(source, binding.imap, site, port)
    return views


def _binding_for(graph, placement, port):
    for binding in graph.bindings:
        for side in (binding.source, binding.target):
            if (
                isinstance(side, (Bundle, MemGrid))
                and side.placement is placement
                and side.port is port
            ):
                return binding
    return None


def _slice_of(source, imap, site, port):
    if not isinstance(source, (Tensor, TensorView)):
        raise SPMWBindingError(
            f"`{port.name}` of a placed fabric is bound to {source!r}, which is "
            f"not a tensor; a sub-fabric receives a slice of one."
        )
    if imap is None or not hasattr(imap, "slice_for"):
        raise SPMWBindingError(
            f"`{port.name}` of a placed fabric needs a slice binding -- spmw.shard "
            f"with dim= or the positional identity -- not an element-wise map."
        )
    spans = imap.slice_for(site)
    return TensorView(
        source,
        tuple(start for start, _ in spans),
        tuple(size for _, size in spans),
        site=site,
    )


class _SubIO:
    """The ``io`` a placed fabric's body receives: one slice per declared port."""

    __slots__ = ("_views",)

    def __init__(self, views):
        object.__setattr__(self, "_views", views)

    def __getattr__(self, name):
        views = object.__getattribute__(self, "_views")
        if name in views:
            return views[name]
        raise AttributeError(
            f"a placed fabric's io has no port `{name}`; it declares "
            f"{', '.join(views) or '(none)'}."
        )


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
    """Within a phase, a brick's clients are all readers, or one writer alone.

    A read puts the brick on the binding's source side and a write on its target
    side, so both have to be gathered before the rule can be applied at all.
    """
    clients = {}
    for binding in graph.bindings:
        if isinstance(binding.target, Brick):
            clients.setdefault((binding.phase, binding.target), []).append(
                ("writer", binding)
            )
        if isinstance(binding.source, Brick):
            clients.setdefault((binding.phase, binding.source), []).append(
                ("reader", binding)
            )
    for (phase, brick), uses in clients.items():
        writers = [b for kind, b in uses if kind == "writer"]
        readers = [b for kind, b in uses if kind == "reader"]
        where = f"phase {phase!r}" if phase else "the fabric"
        if len(writers) > 1:
            raise SPMWMemoryError(
                f"`{brick.name}` has {len(writers)} writers in {where}. Within a "
                f"phase a brick's clients must be all readers, or a single writer "
                f"with none. Separate them with spmw.phase(), or bank the brick so "
                f"each writer owns one."
            )
        if writers and readers:
            raise SPMWMemoryError(
                f"`{brick.name}` is written by {writers[0].kind} and read by "
                f"{readers[0].kind} in {where}, so a reader may see it half-filled. "
                f"Order them with spmw.phase(), which lowers to a barrier."
            )


def bind_check(imap, bundle, tensor, where, extent=None, write=False, block=()):
    """Bounds-check a binding's index map over its whole domain."""
    _check_axes_belong(imap, bundle, where)
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


def _check_axes_belong(imap, bundle, where):
    """Every axis in a binding must be one of its own placement's.

    Axes resolve positionally against the site coordinates, so an axis borrowed
    from another placement would silently read this one's coordinates instead of
    the ones it names.
    """
    spec = getattr(imap, "spec", None)
    if spec is None or getattr(imap, "is_lambda", False):
        return
    own = set(bundle.placement.axes)
    for entry in spec:
        if not isinstance(entry, Expr):
            continue
        for axis in entry.axes():
            root = axis
            while root.source is not None:
                root = root.source
            if root not in own:
                raise SPMWBindingError(
                    f"{where}: the map names axis `{axis.name}`, which belongs to "
                    f"another placement. Axes resolve against this binding's own "
                    f"site coordinates, so borrowing one reads the wrong grid."
                )
