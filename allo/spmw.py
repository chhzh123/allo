# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-builtin, too-few-public-methods, unused-argument, global-statement

"""Single-program multiple-work-unit frontend for Allo.

SPMW lets a spatial accelerator be described by writing one *work unit*, declaring
the *topology* that wires copies of it together, and declaring *boundary roles* that
specialize the copies at the edges of the grid. It is an evolution of ``allo.dataflow``:
where dataflow spells out every processing element and every FIFO by hand (``meta_if``
chains over the PID plus absolute grid arithmetic such as ``fifo_A[i, j + 1].put(...)``),
SPMW makes replication, interconnect, and boundary specialization first-class.

This module provides the Python surface and a static topology checker. A topology is a
pure description: a ``grid`` shape plus a ``link`` function that, for each grid point,
maps a local *port* name either to a neighbor (peer form) or to a rendezvous key (key
form). The checker validates a program before any code is generated:

* peer links are symmetric (``east`` at ``(i, j)`` pairs with ``west`` at ``(i, j + 1)``);
* the grid rank matches the topology's dimensionality;
* every key-form channel has exactly one source and one sink;
* roles, stream flows, and work-unit/role bodies only reference declared ports.

``"src"`` and ``"sink"`` are reserved as key-form direction tags, so a peer link cannot use them
as peer-port names.
"""

import ast
import inspect
import itertools
import textwrap
import warnings

__all__ = [
    "SPMWError",
    "Topology",
    "Grid",
    "mesh",
    "ring",
    "scatter",
    "gather",
    "unit",
    "region",
    "map",
    "stream_in",
    "stream_out",
    "channel",
    "shared",
    "banked",
    "view",
    "place",
    "LogicalBuffer",
    "PortContext",
    "PortHandle",
    "validate",
    "role_partition",
    "role_count",
    "resolve_channels",
    "halo_roles",
    "emit_hls",
    "role_body_count",
    "lower",
    "unroll",
    "build",
]


class SPMWError(Exception):
    """Raised when an SPMW program or topology is statically invalid."""


# A key-form link value is ``(key, "src" | "sink")``; a peer-form link value is
# ``(peer_coordinate, peer_port_name)``. The direction string disambiguates the two.
_SRC = "src"
_SINK = "sink"
_DIRECTIONS = (_SRC, _SINK)


def _is_key_form(value):
    return isinstance(value, tuple) and len(value) == 2 and value[1] in _DIRECTIONS


class Topology:
    """A grid of work units plus the links that connect them.

    Parameters
    ----------
    grid:
        The extent of the grid along each dimension, e.g. ``(rows, cols)``.
    link:
        A callable taking one integer per grid dimension and returning a dict mapping
        a local port name to either ``(peer_coordinate, peer_port)`` (peer form) or
        ``(channel_key, "src" | "sink")`` (key form). ``None`` means replication only,
        with no interconnect.
    """

    def __init__(self, grid, link=None):
        if not isinstance(grid, (tuple, list)) or len(grid) == 0:
            raise SPMWError(f"grid must be a non-empty shape; got {grid!r}")
        self.grid = tuple(int(n) for n in grid)
        if any(n <= 0 for n in self.grid):
            raise SPMWError(f"grid extents must be positive; got {self.grid}")
        self._link = link

    @property
    def dims(self):
        """The dimensionality of the grid (its rank)."""
        return len(self.grid)

    def coords(self):
        """Yield every grid coordinate as a tuple."""
        return itertools.product(*(range(n) for n in self.grid))

    def in_bounds(self, coord):
        """Whether ``coord`` lies inside the grid."""
        return len(coord) == self.dims and all(
            0 <= c < n for c, n in zip(coord, self.grid)
        )

    def links_at(self, coord):
        """The link dict the topology declares at ``coord``."""
        if self._link is None:
            return {}
        try:
            result = self._link(*coord)
        except TypeError as exc:
            raise SPMWError(
                f"topology link expects {self.dims} coordinate argument(s) to match "
                f"grid rank {self.dims}, but invoking it failed: {exc}"
            ) from exc
        if not isinstance(result, dict):
            raise SPMWError(
                "topology link must return a dict mapping port name to a peer or key; "
                f"got {type(result).__name__}"
            )
        return result

    def port_names(self):
        """The union of declared port names across the whole grid."""
        names = set()
        for coord in self.coords():
            names.update(self.links_at(coord).keys())
        return names

    def context(self, coord=None):
        """A work-unit context bound to this topology (optionally placed at ``coord``)."""
        return PortContext(self, coord)

    def boundary_ports_at(self, coord):
        """Peer-form ports at ``coord`` whose peer falls outside the grid."""
        boundary = set()
        for port, target in self.links_at(coord).items():
            if _is_key_form(target):
                continue
            peer_coord, _ = self._parse_peer(port, target)
            if not self.in_bounds(peer_coord):
                boundary.add(port)
        return boundary

    def _parse_peer(self, port, target):
        if not (isinstance(target, tuple) and len(target) == 2):
            raise SPMWError(
                f"port {port!r}: a peer link must be (peer_coordinate, peer_port); "
                f"got {target!r}"
            )
        peer_coord, peer_port = target
        if not isinstance(peer_coord, (tuple, list)):
            raise SPMWError(
                f"port {port!r}: peer coordinate must be a tuple; got {peer_coord!r}"
            )
        peer_coord = tuple(peer_coord)
        if len(peer_coord) != self.dims:
            raise SPMWError(
                f"port {port!r}: peer coordinate {peer_coord} has rank "
                f"{len(peer_coord)}, but the grid rank is {self.dims}"
            )
        if not isinstance(peer_port, str):
            raise SPMWError(
                f"port {port!r}: peer port name must be a string; got {peer_port!r}"
            )
        return peer_coord, peer_port

    def validate(self, external_srcs=frozenset(), external_sinks=frozenset()):
        """Run every static check; raise :class:`SPMWError` on the first violation.

        ``external_srcs``/``external_sinks`` are key-form channel keys whose otherwise-open endpoint
        a region-level ``stream_in``/``stream_out`` supplies (the boundary lanes of a permutation
        network); each such key counts as one extra source/sink so a boundary channel is a valid
        1->1 peer rather than a dangling half-open key.
        """
        self._validate_grid_rank()
        self._validate_peer_symmetry()
        self._validate_key_channels(external_srcs, external_sinks)
        return self

    def _validate_grid_rank(self):
        representative = tuple(0 for _ in self.grid)
        for port, target in self.links_at(representative).items():
            if _is_key_form(target):
                continue
            self._parse_peer(port, target)

    def _validate_peer_symmetry(self):
        for coord in self.coords():
            for port, target in self.links_at(coord).items():
                if _is_key_form(target):
                    continue
                peer_coord, peer_port = self._parse_peer(port, target)
                if not self.in_bounds(peer_coord):
                    # An out-of-range peer marks a boundary port; it drives a role or an
                    # auto-halo loader/drain and needs no reciprocal.
                    continue
                peer_targets = self.links_at(peer_coord)
                if peer_port not in peer_targets:
                    raise SPMWError(
                        f"asymmetric link: {coord}.{port} -> "
                        f"{peer_coord}.{peer_port}, but {peer_coord} declares no port "
                        f"{peer_port!r}"
                    )
                back = peer_targets[peer_port]
                if _is_key_form(back):
                    raise SPMWError(
                        f"asymmetric link: {coord}.{port} points to peer-form "
                        f"{peer_coord}.{peer_port}, which is a key-form port"
                    )
                back_coord, back_port = self._parse_peer(peer_port, back)
                if back_coord != tuple(coord) or back_port != port:
                    raise SPMWError(
                        f"asymmetric link: {coord}.{port} -> "
                        f"{peer_coord}.{peer_port}, but {peer_coord}.{peer_port} -> "
                        f"{back_coord}.{back_port} (expected it to point back to "
                        f"{tuple(coord)}.{port})"
                    )

    def _validate_key_channels(
        self, external_srcs=frozenset(), external_sinks=frozenset()
    ):
        # classifies every key channel and rejects unsupported cardinality
        self.key_channel_roles(external_srcs, external_sinks)

    def key_channel_roles(self, external_srcs=frozenset(), external_sinks=frozenset()):
        """Classify each key channel as ``"peer"`` (1->1), ``"scatter"`` (1->N), or ``"gather"``.

        A scatter is one source fanning out to many sinks; a gather is many sources fanning in to one
        sink; a peer channel is the 1->1 case. A key with no source, no sink, or a many-to-many
        (N->M with both > 1) topology is rejected. A key in ``external_srcs``/``external_sinks`` gains
        a virtual source/sink -- the open boundary lanes a region ``stream_in``/``stream_out`` feeds
        or drains -- so a boundary channel is a valid 1->1 peer.
        """
        sources, sinks = {}, {}
        for coord in self.coords():
            for port, target in self.links_at(coord).items():
                if not _is_key_form(target):
                    continue
                key, direction = target
                bucket = sources if direction == _SRC else sinks
                bucket.setdefault(key, []).append((coord, port))
        roles = {}
        for key in set(sources) | set(sinks) | set(external_srcs) | set(external_sinks):
            n_src = len(sources.get(key, [])) + (1 if key in external_srcs else 0)
            n_sink = len(sinks.get(key, [])) + (1 if key in external_sinks else 0)
            if n_src == 1 and n_sink == 1:
                roles[key] = "peer"
            elif n_src == 1 and n_sink >= 1:
                roles[key] = "scatter"
            elif n_sink == 1 and n_src >= 1:
                roles[key] = "gather"
            else:
                raise SPMWError(
                    f"channel key {key!r} must be one source to one-or-more sinks (scatter) or "
                    f"one-or-more sources to one sink (gather); got {n_src} source(s) and "
                    f"{n_sink} sink(s)"
                )
        return roles

    def __repr__(self):
        return f"Topology(grid={self.grid})"


class Grid(Topology):
    """A replication-only grid with no declared interconnect."""

    def __init__(self, shape):
        super().__init__(grid=shape, link=None)


def mesh(shape):
    """A nearest-neighbor mesh topology (1-D chain or 2-D grid)."""
    shape = tuple(shape)
    if len(shape) == 2:

        def link_2d(i, j):
            return {
                "east": ((i, j + 1), "west"),
                "west": ((i, j - 1), "east"),
                "south": ((i + 1, j), "north"),
                "north": ((i - 1, j), "south"),
            }

        return Topology(grid=shape, link=link_2d)
    if len(shape) == 1:

        def link_1d(i):
            return {"next": ((i + 1,), "prev"), "prev": ((i - 1,), "next")}

        return Topology(grid=shape, link=link_1d)
    raise SPMWError(f"mesh currently supports 1-D and 2-D grids; got rank {len(shape)}")


def ring(n):
    """A 1-D ring topology (a chain with wrap-around)."""
    n = int(n)

    def link(i):
        return {
            "next": (((i + 1) % n,), "prev"),
            "prev": (((i - 1) % n,), "next"),
        }

    return Topology(grid=(n,), link=link)


def scatter(n, key="scatter"):
    """A scatter (fan-out) collective: unit 0 is the source of ``key``; units 1..n-1 are its sinks."""
    n = int(n)

    def link(i):
        return {"out": (key, _SRC)} if i == 0 else {"in": (key, _SINK)}

    return Topology(grid=(n,), link=link)


def gather(n, key="gather"):
    """A gather (fan-in) collective: units 0..n-2 are sources of ``key``; unit n-1 is its one sink."""
    n = int(n)

    def link(i):
        return {"in": (key, _SINK)} if i == n - 1 else {"out": (key, _SRC)}

    return Topology(grid=(n,), link=link)


class PortHandle:
    """A handle to one port of a placed work unit.

    ``put``/``get`` are realized when an SPMW program is lowered; on the bare surface
    the handle exists only so the topology can bind and validate port names.
    """

    def __init__(self, name):
        self.name = name

    def _unrealized(self):
        raise NotImplementedError(
            f"port {self.name!r} I/O is realized during SPMW lowering"
        )

    def put(self, *args, **kwargs):
        self._unrealized()

    def get(self, *args, **kwargs):
        self._unrealized()

    def get_or(self, *args, **kwargs):
        self._unrealized()

    def __repr__(self):
        return f"PortHandle({self.name!r})"


class PortContext:
    """The ``ctx`` handed to a work-unit body: its rank plus its ports."""

    def __init__(self, topology, coord=None):
        self._topology = topology
        self._coord = tuple(coord) if coord is not None else None
        self._ports = topology.port_names()

    def rank(self):
        """This unit's grid coordinate (a scalar for a 1-D grid, else a tuple)."""
        if self._coord is None:
            raise SPMWError("ctx.rank() is only defined for a placed work unit")
        return self._coord[0] if len(self._coord) == 1 else self._coord

    def port(self, name):
        """The handle for a declared port; error on an undeclared name."""
        if name not in self._ports:
            raise SPMWError(
                f"port {name!r} is not declared in the topology; declared ports: "
                f"{sorted(self._ports)}"
            )
        return PortHandle(name)

    def __getattr__(self, name):
        # Directional sugar: ctx.east is ctx.port("east"). Attributes that are not
        # declared ports fall through to a normal AttributeError.
        if name.startswith("_"):
            raise AttributeError(name)
        ports = self.__dict__.get("_ports", set())
        if name in ports:
            return PortHandle(name)
        raise AttributeError(
            f"{name!r} is not a declared port; declared ports: {sorted(ports)}"
        )


class Unit:
    """A single program replicated at each grid point, with optional boundary roles."""

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.interior = fn
        # Each role is (edge_names, body); it specializes the units missing those links.
        self.roles = []
        # Each variant is (predicate_expr, body): a compute body selected where the affine
        # indicator ``predicate_expr`` over the grid coordinates (d0, d1, ...) is nonzero. The
        # base @unit body is the default, applied where no variant's predicate holds.
        self.variants = []
        # A unit may carry its own topology (e.g. a permutation network defined with it).
        self.topo = None

    def role(self, *edges):
        """Register a boundary-role body selected by the given missing-link edges."""
        if not edges:
            raise SPMWError("a role requires at least one boundary edge name")

        def register(body):
            self.roles.append((tuple(edges), body))
            return body

        return register

    def variant(self, when):
        """Register a coordinate-selected compute variant, applied where ``when`` is nonzero.

        ``when`` is an affine indicator expression over the grid coordinates ``d0, d1, ...`` (e.g.
        ``"(d0 + d1) mod 2"`` for a checkerboard). The variant body carries a distinct datapath; it
        lowers to its own ``#spmw.role`` with that predicate, so distinct predicate tags stay
        distinct role bodies through partition and HLS emission rather than being merged.
        """
        if not isinstance(when, str) or not when.strip():
            raise SPMWError("a variant requires a non-empty affine predicate string")

        def register(body):
            self.variants.append((when.strip(), body))
            return body

        return register

    def __repr__(self):
        return f"<spmw.unit {self.name!r} roles={[edges for edges, _ in self.roles]}>"


def unit(fn):
    """Decorator marking ``fn`` as the interior work-unit body."""
    return Unit(fn)


class Region:
    """A composition scope: it holds tensors, declares maps, and wires them together."""

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    def __repr__(self):
        return f"<spmw.region {self.name!r}>"

    def __call__(self, *args, **kwargs):
        """Instantiate this region inside another region: inline its spatial content into the caller.

        Invoking a region within an active region body (``mxu(ub, wbuf, psum)``) traces this region's
        body with the caller's operands bound to its parameters, then merges the resulting maps,
        streams, channels, and placements into the enclosing collection -- so a design is written once
        (the sub-region) and instantiated, while the collected form the backends consume stays flat.
        """
        if _active_collector is None:
            raise SPMWError(
                f"region {self.name!r} can only be invoked inside another region body"
            )
        parent = _active_collector
        sub = _collect_body(self, args, kwargs)
        for channel_decl in sub.channels:
            if any(existing.name == channel_decl.name for existing in parent.channels):
                raise SPMWError(
                    f"nested region {self.name!r} redeclares channel {channel_decl.name!r}"
                )
        parent.maps.extend(sub.maps)
        parent.streams.extend(sub.streams)
        parent.channels.extend(sub.channels)
        parent.placements.extend(sub.placements)
        parent.nested.append(self.name)
        parent.nested.extend(sub.nested)


def region():
    """Decorator factory marking a function as an SPMW composition scope."""

    def decorator(fn):
        return Region(fn)

    return decorator


# Every channel defaults to this FIFO depth unless a ``depths={...}`` override says otherwise.
DEFAULT_DEPTH = 2


class _MapDecl:
    """A recorded ``spmw.map`` call: a unit placed over a topology."""

    def __init__(self, work_unit, topology, depths, shard=None, fold=None, unroll=None):
        # `fold`/`unroll` mirror the module-level pass helpers by name (they are the public kwargs)
        # pylint: disable=redefined-outer-name
        self.unit = work_unit
        self.topology = topology
        self.depths = dict(depths) if depths else {}
        # Resolved per-port FIFO depth: the default for every declared port, overridden per the
        # user's request. This materializes the default so callers do not read the raw override.
        self.port_depths = {port: DEFAULT_DEPTH for port in topology.port_names()}
        self.port_depths.update(self.depths)
        # A shard tiles the grid into a 2-level hierarchy: each shard is a `shard` sub-block of the
        # grid, replicated `grid // shard` times. `None` means a flat (unsharded) map.
        self.shard = tuple(shard) if shard else None
        # Per-dim time-multiplex (`fold`) and spatial-unroll (`unroll`) factors, or `None` when the
        # map is not folded/unrolled. Each is a full-rank tuple whose product with the physical grid
        # reconstructs the logical grid; consumed by the folding/banking machinery.
        self.fold = tuple(fold) if fold is not None else None
        self.unroll = tuple(unroll) if unroll is not None else None


class _StreamDecl:
    """A recorded auto-halo ``stream_in``/``stream_out`` declaration."""

    def __init__(self, tensor, work_unit, flow, direction, **extra):
        self.tensor = tensor
        self.unit = work_unit
        self.flow = flow
        self.direction = direction
        self.extra = extra


class _ChannelDecl:
    """A recorded region-level inter-unit channel: a named FIFO one unit puts to and another gets."""

    def __init__(self, name, dtype, depth):
        self.name = name
        self.dtype = dtype
        self.depth = depth


class _PlacementDecl:
    """A recorded ``spmw.place``: a region tensor operand pinned to a logical memory buffer."""

    def __init__(self, tensor, buffer):
        self.tensor = tensor
        self.buffer = buffer


class _RegionCollection:
    """The maps, streams, channels, and memory placements gathered from one pass over a region."""

    def __init__(self):
        self.maps = []
        self.streams = []
        self.channels = []
        self.placements = []
        # Names of sub-regions invoked (inlined) in this region body, in call order -- a region can
        # instantiate another region (``mxu(ub, wbuf, psum)``), which merges the sub-region's spatial
        # content here. Recorded so codegen/tests can see the hierarchy the flat collection came from.
        self.nested = []


# The collector active while a region body is being traced (None outside a trace).
_active_collector = None


def _resolve_topology(work_unit, grid, topo):
    topology = None
    if topo is not None:
        topology = topo
    elif isinstance(grid, Topology):
        topology = grid
    elif getattr(work_unit, "topo", None) is not None:
        topology = work_unit.topo
    if topology is None:
        if grid is None:
            raise SPMWError("spmw.map needs a grid= or topo=")
        topology = Grid(grid)
    if not isinstance(topology, Topology):
        raise SPMWError(
            f"topology must be an spmw.Topology; got {type(topology).__name__}"
        )
    if grid is not None and not isinstance(grid, Topology):
        if tuple(grid) != topology.grid:
            raise SPMWError(
                f"grid shape {tuple(grid)} does not match topology grid {topology.grid}"
            )
    return topology


def _strict_int(value, what):
    """``value`` as a plain Python ``int``, rejecting ``bool`` and non-ints with ``SPMWError``.

    ``bool`` is an ``int`` subclass, so a ``True``/``False`` axis or factor would otherwise slip
    through as ``1``/``0``; a ``float`` would silently truncate. Both are almost certainly mistakes,
    so both raise instead of coercing.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise SPMWError(f"{what} must be an int, got {value!r}")
    return value


def _resolve_map_factors(factors, topology, kind):
    """A full-rank tuple of ``fold``/``unroll`` factors for a map over ``topology``, or ``None``.

    ``factors`` may be a per-axis mapping (``{axis: factor}`` with an int axis or a ``"row"``/``"col"``
    key, axes default to ``1``) or a full-rank list/tuple. Every axis and factor must be a real
    ``int`` (``bool`` and ``float`` are rejected), and every factor must be positive and divide its
    grid extent, so the physical grid ``grid // factors`` is integral. Any malformed input raises
    ``SPMWError`` (never a leaked ``TypeError``/``ValueError``). ``kind`` (``"fold"``/``"unroll"``)
    only sharpens the messages.
    """
    if factors is None:
        return None
    dims, grid = topology.dims, topology.grid
    if isinstance(factors, dict):
        resolved = [1] * dims
        for key, value in factors.items():
            if isinstance(key, str):
                if key not in _BANK_AXES:
                    raise SPMWError(
                        f"{kind} axis {key!r} is not one of {sorted(_BANK_AXES)} or an int"
                    )
                axis = _BANK_AXES[key]
            else:
                axis = _strict_int(key, f"{kind} axis")
            if not 0 <= axis < dims:
                raise SPMWError(
                    f"{kind} axis {axis} is out of range for a rank-{dims} grid"
                )
            resolved[axis] = _strict_int(value, f"{kind} factor")
    elif isinstance(factors, (list, tuple)):
        resolved = [_strict_int(f, f"{kind} factor") for f in factors]
        if len(resolved) != dims:
            raise SPMWError(
                f"{kind} rank {len(resolved)} does not match grid rank {dims}"
            )
    else:
        raise SPMWError(
            f"{kind} must be a dict, list, or tuple; got {type(factors).__name__}"
        )
    for axis, (factor, extent) in enumerate(zip(resolved, grid)):
        if factor <= 0:
            raise SPMWError(f"{kind} factor {factor} on axis {axis} must be positive")
        if extent % factor != 0:
            raise SPMWError(
                f"{kind} factor {factor} on axis {axis} must divide the grid extent {extent}"
            )
    return tuple(resolved)


def map(
    work_unit, grid=None, topo=None, depths=None, shard=None, fold=None, unroll=None
):
    # `fold`/`unroll` mirror the module-level pass helpers by name (they are the public kwargs)
    # pylint: disable=redefined-outer-name
    """Replicate a work unit over a grid, wired by a topology.

    ``shard`` tiles the grid into a 2-level hierarchy: each shard is a ``shard``-shaped sub-block,
    replicated ``grid // shard`` times. Its rank must match the grid and each extent must divide the
    corresponding grid extent, so a ``P0 x P1`` grid with ``shard=(s0, s1)`` nests
    ``(P0/s0) x (P1/s1)`` shards of ``s0 x s1`` units.

    ``fold`` time-multiplexes the map: a factor ``F`` on an axis runs ``F`` logical grid points on
    one physical unit. ``unroll`` spatially unrolls it. Each is a per-axis ``{axis: factor}`` mapping
    or a full-rank list; every factor must be a positive int dividing its grid extent.
    """
    if not isinstance(work_unit, Unit):
        raise SPMWError("spmw.map expects an @spmw.unit as its first argument")
    topology = _resolve_topology(work_unit, grid, topo)
    fold = _resolve_map_factors(fold, topology, "fold")
    unroll = _resolve_map_factors(unroll, topology, "unroll")
    if depths:
        ports = topology.port_names()
        for port in depths:
            if port not in ports:
                raise SPMWError(
                    f"depths references undeclared port {port!r}; declared ports: "
                    f"{sorted(ports)}"
                )
    if shard is not None:
        shard = tuple(int(s) for s in shard)
        if len(shard) != topology.dims:
            raise SPMWError(
                f"shard rank {len(shard)} does not match grid rank {topology.dims}"
            )
        for extent, tile in zip(topology.grid, shard):
            if tile <= 0 or extent % tile != 0:
                raise SPMWError(
                    f"shard {shard} must divide the grid {topology.grid} along every axis"
                )
    decl = _MapDecl(work_unit, topology, depths, shard, fold, unroll)
    if _active_collector is not None:
        _active_collector.maps.append(decl)
    return decl


def stream_in(tensor, into=None, flow=None, **kwargs):
    """Declare an operand that streams into the grid across a flow direction."""
    decl = _StreamDecl(tensor, into, flow, "in", **kwargs)
    if _active_collector is not None:
        _active_collector.streams.append(decl)
    return decl


def stream_out(tensor, from_=None, flow=None, **kwargs):
    """Declare a result that streams out of the grid."""
    decl = _StreamDecl(tensor, from_, flow, "out", **kwargs)
    if _active_collector is not None:
        _active_collector.streams.append(decl)
    return decl


def channel(name, dtype, depth=DEFAULT_DEPTH):
    """Declare a region-level FIFO channel connecting one producer unit to one consumer unit.

    A unit body puts to / gets from it as a named port (``ctx.<name>.put(...)`` /
    ``ctx.<name>.get()``). Unlike ``stream_in``/``stream_out`` (operand flow across one grid), a
    channel links two *different* mapped units, so a region can express a producer/consumer pipeline.
    """
    if _active_collector is not None:
        for existing in _active_collector.channels:
            if existing.name == name:
                raise SPMWError(f"channel {name!r} is declared more than once")
    decl = _ChannelDecl(name, dtype, depth)
    if _active_collector is not None:
        _active_collector.channels.append(decl)
    return decl


def place(tensor, buffer):
    """Pin a region tensor operand to a logical memory buffer (``spmw.shared``/``banked``/``view``).

    ``tensor`` is the operand name (a string) or the operand placeholder; ``buffer`` is a
    :class:`LogicalBuffer`. The resolved ``Memory`` resource (and banking axis) is carried on the
    rolled IR so the target honors it.
    """
    if not isinstance(buffer, LogicalBuffer):
        raise SPMWError("spmw.place expects a spmw.shared/banked/view buffer")
    name = tensor if isinstance(tensor, str) else getattr(tensor, "name", None)
    if not isinstance(name, str):
        raise SPMWError(
            "spmw.place expects a tensor operand name (a string) as its first argument"
        )
    decl = _PlacementDecl(name, buffer)
    if _active_collector is not None:
        _active_collector.placements.append(decl)
    return decl


# The on-chip memory hierarchy: a ``space=`` level names a place in the hierarchy, resolved up front
# to a concrete ``allo.memory.Memory`` resource; ``resource=`` is an explicit escape hatch that pins
# the resource directly.
_SPACE_LEVELS = {"L1": "LUTRAM", "L2": "URAM", "L3": "BRAM"}

# The tensor axis each banking key names.
_BANK_AXES = {"row": 0, "col": 1}


def _resolve_space(space, resource):
    """The concrete ``allo.memory.Memory`` a logical ``space=`` level (or ``resource=``) resolves to."""
    # pylint: disable=import-outside-toplevel
    from .memory import Memory

    if resource is not None:
        if str(resource).upper() not in Memory.VALID_RESOURCE:
            raise SPMWError(
                f"unknown memory resource {resource!r}; must be one of "
                f"{sorted(Memory.VALID_RESOURCE)}"
            )
        return Memory(resource=resource)
    if space is None:
        return Memory(resource="AUTO")
    if space not in _SPACE_LEVELS:
        raise SPMWError(
            f"unknown memory space {space!r}; known levels are {sorted(_SPACE_LEVELS)} "
            f"(or pass resource=)"
        )
    return Memory(resource=_SPACE_LEVELS[space])


class LogicalBuffer:
    """A logical SPMW buffer: a shaped value placed at a memory ``space=`` level.

    ``memory`` is the resolved :class:`allo.memory.Memory` resource; ``bank_axis`` (banked only) is
    the tensor axis the buffer is partitioned along; ``kind`` is ``"shared"``/``"banked"``/``"view"``;
    ``source`` is the buffer a view aliases. The concrete :class:`allo.memory.Layout` is built by the
    consumer once the buffer shape and the grid it maps onto are known.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        dtype,
        memory,
        kind,
        bank_axis=None,
        shape=None,
        source=None,
        banks=None,
        bank_mode="cyclic",
        stride_bit=None,
        on_conflict="error",
        double=False,
    ):
        self.dtype = dtype
        self.memory = memory
        self.kind = kind
        self.bank_axis = bank_axis
        self.shape = shape
        self.source = source
        # Double buffering (ping-pong): the rolled HLS emitter lowers a `double=True` placement to a
        # two-epoch K-tiled GEMM top with two alternating on-chip copies of the operand's tile (the
        # next tile is preloaded while the current one is consumed). It is a throughput/scheduling
        # choice -- functionally transparent -- so it rides the placement IR (`spmw.double_buffers`)
        # and reshapes the generated backend structure rather than changing the computed result.
        self.double = double
        # Partition-function banking (when banks is set): split the banking axis into `banks`
        # physical banks with an F2 swizzle ("cyclic" = idx % banks, "xor" = a conflict-free
        # butterfly swizzle at `stride_bit`), realized as real 2D [banks][depth] storage.
        self.banks = banks
        self.bank_mode = bank_mode
        self.stride_bit = stride_bit
        self.on_conflict = on_conflict

    def __repr__(self):
        return f"<spmw.{self.kind} {self.dtype!r} @ {self.memory!r}>"


def shared(dtype, space=None, resource=None, double=False):
    """A buffer shared across the grid, placed at a logical ``space=`` level (or a ``resource=``).

    ``double=True`` requests ping-pong double buffering: the rolled emitter lowers the placement to a
    two-epoch K-tiled GEMM top with two alternating on-chip copies, filling the next copy while the
    current one is consumed (see :func:`allo.spmw_hls._pingpong_top_cpp`).
    """
    return LogicalBuffer(
        dtype, _resolve_space(space, resource), "shared", double=double
    )


_BANK_MODES = ("cyclic", "xor")


def banked(
    dtype,
    on=None,
    banks=None,
    bank="cyclic",
    stride_bit=None,
    space=None,
    resource=None,
    on_conflict="error",
):
    """A buffer banked (partitioned) along the ``on`` tensor axis, placed at a ``space=`` level.

    With no ``banks=`` the buffer is fully partitioned along ``on`` (one bank per index), as before.
    A *partition function* splits the banking axis into ``banks`` physical banks with an F2 swizzle:
    ``bank="cyclic"`` (``idx % banks``) or ``bank="xor"`` (a conflict-free butterfly swizzle at
    ``stride_bit``). The swizzle is validated by the F2 solver at lower time (a non-conflict-free /
    too-few-banks banking is rejected), and the emitter realizes it as real 2D ``[banks][depth]``
    banked storage. ``banks`` must be a power of two.
    """
    if on is None:
        raise SPMWError(
            "spmw.banked requires on= naming the banking axis (e.g. 'row'/'col' or an int)"
        )
    if isinstance(on, str):
        if on not in _BANK_AXES:
            raise SPMWError(
                f"unknown banking axis {on!r}; use one of {sorted(_BANK_AXES)} or an int"
            )
        axis = _BANK_AXES[on]
    else:
        axis = _strict_int(on, "banking axis")
    if bank not in _BANK_MODES:
        raise SPMWError(f"spmw.banked bank must be one of {_BANK_MODES}; got {bank!r}")
    if on_conflict not in ("error", "serialize"):
        raise SPMWError(
            f"spmw.banked on_conflict must be 'error' or 'serialize'; got {on_conflict!r}"
        )
    if banks is not None:
        banks = _strict_int(banks, "banks")
        if banks <= 0 or (banks & (banks - 1)) != 0:
            raise SPMWError(
                f"spmw.banked banks must be a positive power of two; got {banks}"
            )
    if stride_bit is not None:
        stride_bit = _strict_int(stride_bit, "stride_bit")
    if bank == "xor" and (banks is None or stride_bit is None):
        raise SPMWError(
            "spmw.banked bank='xor' needs banks= and stride_bit= (the butterfly stage bit)"
        )
    return LogicalBuffer(
        dtype,
        _resolve_space(space, resource),
        "banked",
        bank_axis=axis,
        banks=banks,
        bank_mode=bank,
        stride_bit=stride_bit,
        on_conflict=on_conflict,
    )


def view(source, shape=None):
    """A logical re-view (reshape/alias) of a ``shared``/``banked`` buffer sharing its memory."""
    if not isinstance(source, LogicalBuffer):
        raise SPMWError("spmw.view expects a spmw.shared/banked buffer as its source")
    return LogicalBuffer(
        source.dtype, source.memory, "view", shape=shape, source=source
    )


# Flow shorthands map a direction across a mesh to the (entry, exit) port pair it uses.
# pylint: disable-next=consider-using-namedtuple-or-dataclass
_FLOW_PORTS = {
    "W->E": ("west", "east"),
    "E->W": ("east", "west"),
    "N->S": ("north", "south"),
    "S->N": ("south", "north"),
}


class _TensorPlaceholder:
    """Stands in for a region's tensor argument while its body is traced."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<tensor {self.name}>"


def _make_arguments(fn):
    args, kwargs = [], {}
    for name, param in inspect.signature(fn).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        placeholder = _TensorPlaceholder(name)
        if param.kind == param.KEYWORD_ONLY:
            kwargs[name] = placeholder
        else:
            args.append(placeholder)
    return args, kwargs


def _collect_body(program, args, kwargs):
    """Trace ``program``'s body with ``args``/``kwargs`` bound to its parameters into a fresh
    collection, restoring the previously active collector afterward (so nested traces compose).
    """
    global _active_collector
    collection = _RegionCollection()
    previous = _active_collector
    _active_collector = collection
    try:
        program.fn(*args, **kwargs)
    finally:
        _active_collector = previous
    return collection


def _collect(program):
    args, kwargs = _make_arguments(program.fn)
    return _collect_body(program, args, kwargs)


def _find_map(collection, work_unit):
    for decl in collection.maps:
        if decl.unit is work_unit:
            return decl
    return None


# Stream port I/O in a body looks like ``ctx.<port>.put/get/get_or(...)`` or
# ``ctx.port("<port>").put/get/get_or(...)``. The sentinel marks a port name that a body names
# through a non-literal ``ctx.port(expr)``, which cannot be resolved statically.
_STREAM_METHODS = {"put", "get", "get_or"}
_DYNAMIC_PORT = object()


def _ctx_param_name(fn):
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    for param in params:
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            return param.name
    return None


def _receiver_port(receiver, ctx_name):
    # ``ctx.<port>`` -> the port name.
    if (
        isinstance(receiver, ast.Attribute)
        and isinstance(receiver.value, ast.Name)
        and receiver.value.id == ctx_name
    ):
        return receiver.attr
    # ``ctx.port("literal")`` -> the literal, else the dynamic sentinel.
    if (
        isinstance(receiver, ast.Call)
        and isinstance(receiver.func, ast.Attribute)
        and receiver.func.attr == "port"
        and isinstance(receiver.func.value, ast.Name)
        and receiver.func.value.id == ctx_name
    ):
        if (
            len(receiver.args) == 1
            and isinstance(receiver.args[0], ast.Constant)
            and isinstance(receiver.args[0].value, str)
        ):
            return receiver.args[0].value
        return _DYNAMIC_PORT
    return None


def _collect_port_aliases(tree, ctx_name):
    """Local variables bound to a port handle: ``h = ctx.<port>`` or ``h = ctx.port("...")``.

    Maps each such name to the set of port names (or the dynamic sentinel) it was bound to; a name
    bound to anything that is not a port handle never enters the map, so ``d.get(...)`` on an
    unrelated object is left alone.
    """
    aliases = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        port = _receiver_port(node.value, ctx_name)
        if port is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                aliases.setdefault(target.id, set()).add(port)
    return aliases


def _check_body_ports(label, fn, ports):
    """Reject stream I/O on an undeclared port inside a work-unit or role body.

    Bodies are scanned by source, never executed: a real body holds loops and arithmetic that a
    bare ``ctx`` cannot run. When source is unavailable the scan is skipped rather than failing.
    Stream I/O is recognized directly (``ctx.<port>.put(...)`` / ``ctx.port("p").get()``) and
    through a local alias (``h = ctx.<port>`` then ``h.get()``).
    """
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    ctx_name = _ctx_param_name(fn)
    if ctx_name is None:
        return
    aliases = _collect_port_aliases(tree, ctx_name)

    def reject(port):
        if port is _DYNAMIC_PORT:
            raise SPMWError(
                f"{label}: ctx.port(...) needs a string-literal port name so it can be "
                f"statically checked"
            )
        if port not in ports:
            raise SPMWError(
                f"{label}: stream I/O on undeclared port {port!r}; declared ports: "
                f"{sorted(ports)}"
            )

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in _STREAM_METHODS:
            continue
        receiver = node.func.value
        port = _receiver_port(receiver, ctx_name)
        if port is not None:
            reject(port)
        elif isinstance(receiver, ast.Name) and receiver.id in aliases:
            for aliased in aliases[receiver.id]:
                reject(aliased)


def _key_form_keys(topology):
    """The set of rendezvous keys the topology's key-form links use."""
    keys = set()
    for coord in topology.coords():
        for target in topology.links_at(coord).values():
            if _is_key_form(target):
                keys.add(target[0])
    return keys


def _stream_families(stream):
    """The lane-family name(s) a key-stage ``stream_in``/``stream_out`` targets, or ``None``.

    A key-stage stream sets ``into=``/``from_=`` to a lane-family name (a string) or a tuple of names
    -- unlike a mesh stream, whose target is an ``@spmw.unit``. Returns the family tuple, or ``None``
    when the target is not a family (so the mesh-unit path handles it).
    """
    target = stream.unit
    if isinstance(target, str):
        return (target,)
    if isinstance(target, tuple) and target and all(isinstance(t, str) for t in target):
        return target
    return None


def _boundary_keys(topology, streams):
    """``(external_srcs, external_sinks)``: the boundary lane keys the region streams feed/drain.

    A lane key is ``(family, stage, slot)``. A key-stage ``stream_in`` into a family at ``at_stage``
    supplies the source for that family/stage's lane keys (which the topology leaves open at the
    entry edge); a ``stream_out`` supplies the sink at the exit edge. Keys are matched by family and
    stage, so :meth:`Topology.validate` can treat the open boundary lanes as 1->1 peers.
    """
    keys = _key_form_keys(topology)
    ext_src, ext_sink = set(), set()
    for stream in streams:
        families = _stream_families(stream)
        if families is None:
            continue
        stage = stream.extra.get("at_stage")
        for key in keys:
            if (
                isinstance(key, tuple)
                and len(key) >= 2
                and key[0] in families
                and key[1] == stage
            ):
                (ext_src if stream.direction == "in" else ext_sink).add(key)
    return ext_src, ext_sink


def _validate_key_stage_stream(collection, stream, families):
    """Check a key-stage ``stream_in``/``stream_out``: an integer stage and known lane families."""
    stage = stream.extra.get("at_stage")
    if not isinstance(stage, int) or isinstance(stage, bool):
        raise SPMWError(
            f"stream_{stream.direction} into a lane family needs an integer at_stage=; got "
            f"{stage!r}"
        )
    known = set()
    for decl in collection.maps:
        known |= {
            k[0] for k in _key_form_keys(decl.topology) if isinstance(k, tuple) and k
        }
    for family in families:
        if family not in known:
            raise SPMWError(
                f"stream_{stream.direction} references lane family {family!r}, which no mapped "
                f"topology declares; declared families: {sorted(known)}"
            )
    index = stream.extra.get("index")
    if stream.direction == "in" and index is not None and not callable(index):
        raise SPMWError(
            f"stream_in index= must be a callable slot permutation; got {index!r}"
        )


def _validate_collection(collection):
    if not collection.maps:
        raise SPMWError("region declares no spmw.map")
    # A region-level channel is a valid port name in any unit body (a unit puts to / gets from it).
    channel_ports = {ch.name for ch in collection.channels}
    for decl in collection.maps:
        # Key-stage streams feed/drain the topology's open boundary lanes, so pass those keys to the
        # key-channel check (else a stage-0 sink-only / stage-S source-only lane reads as dangling).
        ext_src, ext_sink = _boundary_keys(decl.topology, collection.streams)
        decl.topology.validate(ext_src, ext_sink)
        ports = decl.topology.port_names()
        for edges, _ in decl.unit.roles:
            for edge in edges:
                if edge not in ports:
                    raise SPMWError(
                        f"role on unit {decl.unit.name!r} references undeclared port "
                        f"{edge!r}; declared ports: {sorted(ports)}"
                    )
        # A non-mesh (e.g. 1-D vector) unit names its stream ports with ``as_=`` on the stream_in/out
        # targeting it (``ctx.<as_>.get()``/``.put()``), since it has no mesh flow ports; add those.
        stream_ports = {
            stream.extra["as_"]
            for stream in collection.streams
            if stream.unit is decl.unit and isinstance(stream.extra.get("as_"), str)
        }
        body_ports = ports | channel_ports | stream_ports
        _check_body_ports(f"unit {decl.unit.name!r}", decl.unit.interior, body_ports)
        for edges, body in decl.unit.roles:
            _check_body_ports(
                f"role {edges} on unit {decl.unit.name!r}", body, body_ports
            )
    for stream in collection.streams:
        # Flow spelling is checked regardless of the target so a typo never slips through.
        if stream.flow is not None and stream.flow not in _FLOW_PORTS:
            raise SPMWError(
                f"unknown stream flow {stream.flow!r}; supported flows: "
                f"{sorted(_FLOW_PORTS)}"
            )
        target = stream.unit
        if target is None:
            raise SPMWError(
                f"stream_{stream.direction} needs a target unit (into=/from_=)"
            )
        families = _stream_families(stream)
        if families is not None and stream.flow is None:
            # Key-stage streaming into a lane family (the permutation-network entry/exit edges): it
            # uses at_stage=, not a mesh flow=. A family name given with a flow= is a malformed mesh
            # stream (a unit was expected), so it falls through to the @spmw.unit check below.
            _validate_key_stage_stream(collection, stream, families)
            continue
        if not isinstance(target, Unit):
            raise SPMWError(
                f"stream_{stream.direction} target must be an @spmw.unit (into=/from_=) or a "
                f"lane-family name; got {target!r}"
            )
        decl = _find_map(collection, target)
        if decl is None:
            raise SPMWError(
                f"stream_{stream.direction} targets unit {target.name!r}, which is not "
                f"mapped in this region"
            )
        ports = decl.topology.port_names()
        if stream.flow is not None:
            for port in _FLOW_PORTS[stream.flow]:
                if port not in ports:
                    raise SPMWError(
                        f"stream flow {stream.flow!r} needs port {port!r}, which is not "
                        f"declared in the topology; declared ports: {sorted(ports)}"
                    )
    return collection


def role_partition(topology):
    """Group grid points by their missing-link signature (link-presence classification).

    Two grid points share a role when the same set of links is missing at them. For a mesh, a
    point's missing links are its out-of-range neighbors, so the interior, the four edges, and the
    four corners each form one role — a 2-D mesh with both extents >= 3 partitions into exactly
    nine roles, independent of the grid size, while a degenerate grid (1-D, or an extent < 3)
    yields the correct smaller set. Returns an insertion-ordered dict mapping each signature (a
    sorted tuple of missing port names) to the list of coordinates that share it.
    """
    if not isinstance(topology, Topology):
        raise SPMWError("spmw.role_partition expects an spmw.Topology")
    groups = {}
    for coord in topology.coords():
        signature = tuple(sorted(topology.boundary_ports_at(coord)))
        groups.setdefault(signature, []).append(coord)
    return groups


def role_count(topology):
    """The number of distinct link-presence roles in a topology (the O(#roles) count)."""
    return len(role_partition(topology))


def resolve_channels(topology):
    """Enumerate the distinct channels of a topology, grouped into per-port-pair FIFO families.

    A peer link and its reciprocal are one channel, so directed edges are deduplicated into
    undirected channels (each counted once via its canonical, lexicographically smaller endpoint)
    and grouped by the unordered pair of port names they connect (e.g. ``east/west``). Key-form
    endpoints sharing a key form one channel, grouped under that key. Returns an ordered dict
    mapping a family name to the list of channels ``(src_coord, src_port, sink_coord, sink_port)``.
    The number of families is constant as the grid scales even though the channel count is
    O(P0*P1) — the FIFO arrays HLS declares.
    """
    if not isinstance(topology, Topology):
        raise SPMWError("spmw.resolve_channels expects an spmw.Topology")
    families = {}
    seen = set()
    key_endpoints = {}
    for coord in topology.coords():
        for port, target in topology.links_at(coord).items():
            if _is_key_form(target):
                key, direction = target
                key_endpoints.setdefault(key, {})[direction] = (tuple(coord), port)
                continue
            peer_coord, peer_port = topology._parse_peer(port, target)
            if not topology.in_bounds(peer_coord):
                continue  # a boundary port carries no channel
            here = (tuple(coord), port)
            there = (peer_coord, peer_port)
            channel_id = frozenset((here, there))
            if channel_id in seen:
                continue
            seen.add(channel_id)
            src, sink = sorted((here, there))
            family = "/".join(sorted((port, peer_port)))
            families.setdefault(family, []).append((src[0], src[1], sink[0], sink[1]))
    for key, endpoints in key_endpoints.items():
        if _SRC in endpoints and _SINK in endpoints:
            src, sink = endpoints[_SRC], endpoints[_SINK]
            families.setdefault(f"key:{key!r}", []).append(
                (src[0], src[1], sink[0], sink[1])
            )
    return families


def _hls_role_name(signature):
    return "interior" if not signature else "_".join(signature)


def emit_hls(program):
    """Emit HLS C++ for a nearest-neighbor mesh region: one function body per link-presence role.

    The number of role function bodies is bounded by the role count and stays constant as the grid
    scales -- the synthesis re-schedules once per role, never once per grid point -- with a rolled
    ``#pragma HLS unroll`` instantiation loop over per-key FIFO arrays. The role bodies are
    signature-level here; the datapath and Vitis synthesis are a later step. This demonstrates the
    hard metric (the count of distinct synthesized function bodies), not the wall-clock trend.
    """
    collection = _validate_collection(_collect(program))
    if len(collection.maps) != 1:
        raise SPMWError("HLS emission currently handles exactly one mapped unit")
    topology = collection.maps[0].topology
    ports = topology.port_names()
    if ports != {"east", "west", "north", "south"}:
        raise SPMWError("HLS emission currently handles a 2-D nearest-neighbor mesh")
    stream = "hls::stream<float>"
    signature = ", ".join(f"{stream} &{port}" for port in sorted(ports))

    roles = role_partition(topology)
    bodies = [
        f"// role {_hls_role_name(sig)}: {len(points)} instance(s)\n"
        f"void {program.name}_pe_{_hls_role_name(sig)}({signature}) {{\n"
        f"  // work-unit datapath\n"
        f"}}"
        for sig, points in roles.items()
    ]

    rows, cols = topology.grid
    fifos = [
        f"  {stream} fifo_{family.replace('/', '_')}[{rows + 1}][{cols + 1}];"
        for family in resolve_channels(topology)
    ]
    call = ", ".join(
        {
            "west": "fifo_east_west[i][j]",
            "east": "fifo_east_west[i][j + 1]",
            "north": "fifo_north_south[i][j]",
            "south": "fifo_north_south[i + 1][j]",
        }[port]
        for port in sorted(ports)
    )
    top = (
        f"void {program.name}_top() {{\n"
        f"#pragma HLS dataflow\n" + "\n".join(fifos) + "\n"
        f"  for (int i = 0; i < {rows}; i++) {{\n"
        f"    #pragma HLS unroll\n"
        f"    for (int j = 0; j < {cols}; j++) {{\n"
        f"      #pragma HLS unroll\n"
        f"      {program.name}_pe_interior({call});\n"
        f"    }}\n"
        f"  }}\n"
        f"}}"
    )
    return "#include <hls_stream.h>\n\n" + "\n\n".join(bodies) + "\n\n" + top


def role_body_count(program):
    """The number of distinct HLS role function bodies emitted for a region (the O(#roles) count)."""
    collection = _validate_collection(_collect(program))
    if len(collection.maps) != 1:
        raise SPMWError("role_body_count currently handles exactly one mapped unit")
    return len(role_partition(collection.maps[0].topology))


def halo_roles(program):
    """The boundary roles synthesized from a region's auto-halo stream declarations.

    A ``stream_in`` with a flow direction makes an operand stream across the array: units on the
    entry edge load it (a ``"loader"`` role), units on the exit edge drain it (a ``"drain"`` role),
    and interior units forward it via the unit body; a ``stream_out`` with a flow collects at its
    exit edge. Returns a list of ``(edge_port, kind, tensor_name)`` for each synthesized boundary
    role, so the loaders/drains are derived from the declared flow rather than hand-written.
    """
    collection = _validate_collection(_collect(program))
    roles = []
    for stream in collection.streams:
        if stream.flow is None:
            continue
        entry, exit_edge = _FLOW_PORTS[stream.flow]
        name = getattr(stream.tensor, "name", str(stream.tensor))
        if stream.direction == "in":
            roles.append((entry, "loader", name))
            roles.append((exit_edge, "drain", name))
        else:
            roles.append((exit_edge, "collector", name))
    return roles


def validate(program):
    """Trace a region and run every static check; return the gathered declarations."""
    if not isinstance(program, Region):
        raise SPMWError("spmw.validate expects an @spmw.region")
    return _validate_collection(_collect(program))


def _translation_offset(topology, port):
    """The constant ``peer - self`` offset of a peer-form port across the whole grid.

    Returns the offset tuple for a pure-translation link (meshes, chains), or ``None`` if the
    port is key-form or its offset is not constant (e.g. a cyclic ring wraps at the boundary).
    """
    offset = None
    for coord in topology.coords():
        target = topology.links_at(coord).get(port)
        if target is None or _is_key_form(target):
            return None
        peer_coord, _ = topology._parse_peer(port, target)
        here = tuple(p - c for p, c in zip(peer_coord, coord))
        if offset is None:
            offset = here
        elif here != offset:
            return None
    return offset


def _affine_map_text(offset):
    dims = len(offset)
    ins = ", ".join(f"d{k}" for k in range(dims))
    outs = []
    for k, step in enumerate(offset):
        if step == 0:
            outs.append(f"d{k}")
        elif step > 0:
            outs.append(f"d{k} + {step}")
        else:
            outs.append(f"d{k} - {-step}")
    return f"affine_map<({ins}) -> ({', '.join(outs)})>"


def _roles_of(decl):
    """The compute and boundary roles for a mapped unit, each as ``(name, missing, predicate, body)``.

    The interior body is the default compute role (no predicate); each declared variant is an extra
    compute role over the same (empty) missing set, selected by its coordinate predicate; each
    declared boundary role specializes the units missing its edges.
    """
    roles = [("interior", [], None, decl.unit.interior)]
    for idx, (when, body) in enumerate(decl.unit.variants):
        roles.append((f"variant{idx}", [], when, body))
    for edges, body in decl.unit.roles:
        roles.append(("_".join(edges), list(edges), None, body))
    return roles


def _predicate_map_text(decl, when):
    """An ``affine_map<(d0, d1, ...) -> (when)>`` over the grid coordinates for a variant predicate."""
    dims = ", ".join(f"d{i}" for i in range(decl.topology.dims))
    return f"affine_map<({dims}) -> ({when})>"


def _topology_text(decl):
    topology = decl.topology
    rep = tuple(0 for _ in topology.grid)
    links = []
    for port, target in sorted(topology.links_at(rep).items()):
        depth = decl.port_depths.get(port, DEFAULT_DEPTH)
        if _is_key_form(target):
            # A key-form link rendezvous by key: the rolled IR names the channel-group *family*
            # (`key[0]` of a `(family, stage, slot)` key, or the bare string key itself) -- the
            # O(#roles) interconnect identity, exactly as a peer link names "east/west" rather than
            # every concrete FIFO. The per-instance stage/slot permutation is a datapath/layout
            # detail carried separately (and checked), not a rolled representative-point link.
            key, endpoint = target
            family = key[0] if isinstance(key, tuple) and key else key
            if not isinstance(family, str):
                raise SPMWError(
                    f"port {port!r}: key-form channel family must be a string; got {family!r}"
                )
            links.append(
                f'#spmw.key_link<port = "{port}", key = "{family}", '
                f'end = "{endpoint}", depth = {depth}>'
            )
            continue
        offset = _translation_offset(topology, port)
        if offset is None:
            raise SPMWError(
                f"port {port!r}: only affine-translation peer links are lowerable so far"
            )
        _, peer_port = topology._parse_peer(port, target)
        links.append(
            f'#spmw.peer_link<port = "{port}", map = {_affine_map_text(offset)}, '
            f'peer = "{peer_port}", depth = {depth}>'
        )
    grid_text = ", ".join(str(n) for n in topology.grid)
    if links:
        links_text = "[\n      " + ",\n      ".join(links) + "\n    ]"
    else:
        links_text = "[]"
    return (
        f"#spmw.topology<grid = [{grid_text}], dims = {topology.dims}, "
        f"links = {links_text}>"
    )


def _interior_ports():
    """The local ports the transcribed interior func streams over, in signature (sorted) order."""
    # pylint: disable=import-outside-toplevel
    from .spmw_mlir import _READ_PORTS

    return sorted(_READ_PORTS)


def _interior_role_func(program, decl, sym, body=None):
    """The compute role ``func.func`` with the real transcribed datapath, or ``None``.

    ``body`` is the work-unit body to transcribe (the interior body by default, or a
    predicate-selected variant body). ``None`` is returned when the region is not the systolic
    A/B/C operand shape the datapath transcriber handles (e.g. the minimal regions used by the
    dialect tests), so the caller falls back to the signature-only stub.
    """
    # pylint: disable=import-outside-toplevel
    from .spmw_mlir import interior_role_func

    annotations = getattr(program.fn, "__annotations__", {})
    tensors = [v for v in annotations.values() if getattr(v, "shape", None)]
    if len(tensors) < 3:
        return None
    shapes = (tensors[0].shape, tensors[1].shape, tensors[2].shape)
    # The datapath transcriber handles only the 2-D systolic A/B/C operand shape. A region whose
    # first three operands are not all 2-D (e.g. the 1-D FFT lane vectors, or a key-form topology)
    # falls back to the signature-only stub role func rather than crashing on `shapes[0][1]`.
    if any(len(shape) != 2 for shape in shapes):
        return None
    elem = repr(tensors[0].dtype)
    trip = shapes[0][1]  # A is [M, K]; the unit's k-loop runs K times
    # A systolic region fails closed: an untranscribable interior body raises rather than silently
    # becoming an empty stub. (Genuinely topology-only regions already returned None above.)
    return interior_role_func(
        sym, body or decl.unit.interior, elem, trip, decl.port_depths, shapes
    )


def _stream_operand_info(program, stream):
    """``(index, ssa, shape, elem, tag)`` for the tensor a ``stream_in`` declares.

    Derived from the streamed tensor's identity (``stream.tensor.name``) and its position among the
    region's shaped operands -- not from the flow direction -- so the halo operand tracks the actual
    declaration rather than an A/B position convention.
    """
    name = getattr(stream.tensor, "name", None)
    annotations = getattr(program.fn, "__annotations__", {})
    index = 0
    for pname, typ in annotations.items():
        if not getattr(typ, "shape", None):
            continue
        if pname == name:
            return index, f"%{pname}", typ.shape, repr(typ.dtype), pname
        index += 1
    raise SPMWError(
        f"stream_in tensor {name!r} is not a shaped operand of region {program.name!r}"
    )


def _halo_role_funcs(program, decl, collection):
    """Synthesize the loader/drain role funcs implied by the region's stream flows.

    Each ``stream_in`` with a flow makes an operand stream across the array: a loader at the entry
    edge reads the operand row/col and streams it in, a drain at the exit edge consumes it. These
    are boundary tasks *around* the compute grid, not compute-grid roles -- an edge PE still runs
    the interior compute -- so they are emitted as separate funcs and are never assigned to a grid
    point. Returns a list of func texts, empty when the region is not the two-operand systolic shape
    the datapath handles.
    """
    # pylint: disable=import-outside-toplevel
    from .spmw_mlir import drain_role_func, loader_role_func

    annotations = getattr(program.fn, "__annotations__", {})
    tensors = [v for v in annotations.values() if getattr(v, "shape", None)]
    if len(tensors) < 3:
        return []
    prefix = f"{program.name}_{decl.unit.name}"
    out = []
    for stream in collection.streams:
        if stream.direction != "in" or stream.flow is None:
            continue
        entry, exit_edge = _FLOW_PORTS[stream.flow]
        horizontal = stream.flow in {"W->E", "E->W"}
        _, operand_ssa, shape, elem, tag = _stream_operand_info(program, stream)
        trip = (
            shape[1] if horizontal else shape[0]
        )  # the streamed (contraction) dimension
        # each halo task's stream must match the depth of the boundary port it declares (the loader
        # feeds the entry edge, the drain consumes the exit edge), so a per-family depth flows through
        load_depth = decl.port_depths.get(entry, DEFAULT_DEPTH)
        drain_depth = decl.port_depths.get(exit_edge, DEFAULT_DEPTH)
        load_sym, drain_sym = f"{prefix}_load_{tag}", f"{prefix}_drain_{tag}"
        out.append(
            loader_role_func(
                load_sym,
                elem,
                trip,
                load_depth,
                operand_ssa,
                shape,
                exit_edge,
                horizontal,
            )
        )
        out.append(drain_role_func(drain_sym, elem, trip, drain_depth, entry))
    return out


def _halo_tasks(program, decl, collection):
    """The ``#spmw.halo`` descriptors for a mapped unit's loader/drain boundary tasks.

    Mirrors :func:`_halo_role_funcs`: each ``stream_in`` flow makes a loader at the entry edge and a
    drain at the exit edge. The loader feeds the entry port of every point missing that port (the
    entry edge), reading the operand indexed by the coordinate along ``axis``; the drain consumes the
    exit port of every point missing it. ``spmw-unroll`` reads these and wires each task to the same
    boundary channel the edge point's compute role uses.
    """
    annotations = getattr(program.fn, "__annotations__", {})
    tensors = [v for v in annotations.values() if getattr(v, "shape", None)]
    if len(tensors) < 3:
        return []
    prefix = f"{program.name}_{decl.unit.name}"
    tasks = []
    for stream in collection.streams:
        if stream.direction != "in" or stream.flow is None:
            continue
        entry, exit_edge = _FLOW_PORTS[stream.flow]
        horizontal = stream.flow in {"W->E", "E->W"}
        operand, _, _, _, tag = _stream_operand_info(program, stream)
        axis = (
            0 if horizontal else 1
        )  # horizontal loaders index by row, vertical by column
        load_sym, drain_sym = f"{prefix}_load_{tag}", f"{prefix}_drain_{tag}"
        tasks.append(
            f'#spmw.halo<unit = @{load_sym}, port = "{entry}", kind = "load", '
            f"axis = {axis}, operand = {operand}>"
        )
        tasks.append(
            f'#spmw.halo<unit = @{drain_sym}, port = "{exit_edge}", kind = "drain", '
            f"axis = {axis}, operand = {operand}>"
        )
    return tasks


def _channel_endpoints(program, collection):
    """``{channel_name: (producer_sym, consumer_sym)}`` -- the role funcs that put to / get from each.

    Endpoints are counted per mapped declaration: a channel must have exactly one distinct producer
    map (puts) and one distinct consumer map (gets), and they must differ (no self-loop). This
    mirrors the simulator path's :func:`spmw_datapath._validate_pipeline_channels`, so the rolled IR
    never records an invalid channel topology. The symbol is the producing/consuming map's interior
    role func.
    """
    channel_names = {ch.name for ch in collection.channels}
    producers = {name: {} for name in channel_names}  # name -> {map index: sym}
    consumers = {name: {} for name in channel_names}
    for idx, decl in enumerate(collection.maps):
        sym = f"{program.name}_{decl.unit.name}_interior"
        try:
            source = textwrap.dedent(inspect.getsource(decl.unit.interior))
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            continue
        ctx_name = _ctx_param_name(decl.unit.interior)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ):
                continue
            name = _receiver_port(node.func.value, ctx_name)
            if name not in channel_names:
                continue
            if node.func.attr == "put":
                producers[name][idx] = sym
            elif node.func.attr in {"get", "get_or"}:
                consumers[name][idx] = sym
    endpoints = {}
    for name in channel_names:
        prod, cons = producers[name], consumers[name]
        if len(prod) != 1 or len(cons) != 1:
            raise SPMWError(
                f"channel {name!r} must have exactly one producer map and one consumer map; got "
                f"{len(prod)} producer(s) and {len(cons)} consumer(s)"
            )
        if set(prod) == set(cons):
            raise SPMWError(
                f"channel {name!r} is put and got by the same map (self-loop); a channel connects "
                f"two different maps"
            )
        endpoints[name] = (next(iter(prod.values())), next(iter(cons.values())))
    return endpoints


def _memory_attrs(program, collection):
    """``#spmw.memory`` attrs for the region's ``spmw.place`` buffers, carrying the resolved resource.

    Each placement pins a region tensor operand to a logical buffer; the resolved ``Memory``
    resource (and banking axis, ``-1`` when unbanked) rides on the rolled IR. A placement on an
    operand the region does not declare is rejected.
    """
    annotations = getattr(program.fn, "__annotations__", {})
    tensor_names = {
        name for name, typ in annotations.items() if getattr(typ, "shape", None)
    }
    attrs = []
    for decl in collection.placements:
        if decl.tensor not in tensor_names:
            raise SPMWError(
                f"spmw.place target {decl.tensor!r} is not a shaped operand of region "
                f"{program.name!r}"
            )
        bank_axis = decl.buffer.bank_axis if decl.buffer.bank_axis is not None else -1
        attrs.append(
            f'#spmw.memory<tensor = "{decl.tensor}", '
            f'resource = "{decl.buffer.memory.resource}", bank_axis = {bank_axis}>'
        )
    return attrs


def _double_buffer_attrs(program, collection):
    """The tensor names of ``spmw.place`` buffers requesting ping-pong double buffering.

    Carried on a plain ``spmw.double_buffers`` string list (a builtin ArrayAttr) alongside the typed
    ``#spmw.memory`` placements, so the ``double=True`` request survives to codegen without extending
    the ``spmw`` dialect. The rolled emitter consumes it to emit a two-copy ping-pong GEMM top
    (:func:`allo.spmw_hls._pingpong_top_cpp`).
    """
    annotations = getattr(program.fn, "__annotations__", {})
    tensor_names = {
        name for name, typ in annotations.items() if getattr(typ, "shape", None)
    }
    return [
        f'"{decl.tensor}"'
        for decl in collection.placements
        if decl.tensor in tensor_names and getattr(decl.buffer, "double", False)
    ]


def _bank_function_attrs(program, collection):
    """``spmw.bank_functions`` entries for placed banked buffers carrying an F2 partition function.

    Each entry ``"<tensor>:<banks>:<stride_bit>:<mode>"`` names a real F2-banked storage layout the
    emitter realizes as a 2D ``[banks][depth]`` array. The swizzle is **validated by the F2 solver
    here** — a non-conflict-free or too-few-banks banking raises (or, under ``on_conflict="serialize"``,
    falls back to a single reported bank) — so nothing unrealizable reaches the emitter. A plain
    banked buffer (no ``banks=``) emits no entry, so unbanked designs are byte-identical.
    """
    # pylint: disable=import-outside-toplevel
    from .transform.f2_layout import F2LayoutSolver

    annotations = getattr(program.fn, "__annotations__", {})
    shapes = {
        name: typ.shape
        for name, typ in annotations.items()
        if getattr(typ, "shape", None)
    }
    attrs = []
    for decl in collection.placements:
        buffer = decl.buffer
        if buffer.kind != "banked" or buffer.banks is None:
            continue
        if decl.tensor not in shapes:
            continue
        shape = shapes[decl.tensor]
        axis = buffer.bank_axis
        if not 0 <= axis < len(shape):
            raise SPMWError(
                f"banked buffer for {decl.tensor!r} banks axis {axis}, out of range for its "
                f"shape {tuple(shape)}"
            )
        size = int(shape[axis])
        if size & (size - 1) != 0:
            raise SPMWError(
                f"banked buffer for {decl.tensor!r}: F2 banking needs a power-of-two banking-axis "
                f"size, got {size}"
            )
        if buffer.banks > size:
            raise SPMWError(
                f"banked buffer for {decl.tensor!r} has banks {buffer.banks} > banking-axis "
                f"size {size}"
            )
        n_bits = size.bit_length() - 1
        bank_bits = buffer.banks.bit_length() - 1
        stride = buffer.stride_bit if buffer.bank_mode == "xor" else None
        try:
            F2LayoutSolver(n_bits, bank_bits).solve(
                [stride] if stride is not None else []
            )
        except ValueError as err:
            if buffer.on_conflict == "serialize":
                warnings.warn(
                    f"spmw.banked for {decl.tensor!r} is not conflict-free ({err}); serializing "
                    f"to a single bank as requested (on_conflict='serialize')",
                    stacklevel=2,
                )
                attrs.append(f'"{decl.tensor}:1:0:cyclic:{axis}"')
                continue
            raise SPMWError(
                f"spmw.banked for {decl.tensor!r} is not a conflict-free banking: {err}"
            ) from err
        attrs.append(
            f'"{decl.tensor}:{buffer.banks}:{stride if stride is not None else 0}:'
            f'{buffer.bank_mode}:{axis}"'
        )
    return attrs


def _channel_attrs(program, collection):
    """``#spmw.channel`` attrs for the region's inter-unit channels, preserving them in the rolled IR.

    A channel connects two distinct maps, so it is carried on the top function (not on a single map);
    an M2 pass reads these to resolve/lower the pipeline. A shaped payload becomes a ``memref`` type;
    ``producer``/``consumer`` name the role funcs that put to / get from it.
    """
    endpoints = _channel_endpoints(program, collection)
    attrs = []
    for ch in collection.channels:
        shape = getattr(ch.dtype, "shape", None)
        if shape:
            dims = "x".join(str(d) for d in shape)
            elem = f"memref<{dims}x{repr(ch.dtype.dtype)}>"
        else:
            elem = repr(ch.dtype)
        producer_sym, consumer_sym = endpoints[
            ch.name
        ]  # validated non-empty + distinct
        attrs.append(
            f'#spmw.channel<name = "{ch.name}", type = {elem}, depth = {ch.depth}, '
            f"producer = @{producer_sym}, consumer = @{consumer_sym}>"
        )
    return attrs


def _tensor_operands(program):
    """The region's shaped memref args as ``(ssa, memref type)`` pairs -- the map's ``$tensors``."""
    annotations = getattr(program.fn, "__annotations__", {})
    operands = []
    for name, typ in annotations.items():
        if getattr(typ, "shape", None):
            dims = "x".join(str(d) for d in typ.shape)
            operands.append((f"%{name}", f"memref<{dims}x{repr(typ.dtype)}>"))
    return operands


def _role_body_is_trivial(body):
    """Whether an ``@pe.role`` body is empty (only ``pass`` / a docstring) and so needs no datapath."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(body)))
    stmts = tree.body[0].body
    meaningful = [
        node
        for node in stmts
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    return all(isinstance(node, ast.Pass) for node in meaningful)


def _check_role_bodies_transcribable(decl):
    """Fail closed on an explicit role whose body carries real work we cannot transcribe yet.

    Explicit ``@pe.role`` bodies currently lower to signature-only stubs. A non-empty role body would
    therefore be *silently dropped*, so it is rejected instead: only trivial (``pass``) role bodies
    lower to a stub. (Transcribing explicit role datapaths is a later step and needs a declared
    element type, which topology-only regions do not have.)
    """
    for edges, body in decl.unit.roles:
        if not _role_body_is_trivial(body):
            raise SPMWError(
                f"explicit role {list(edges)} has a body spmw.lower cannot transcribe yet; "
                f"only trivial (pass) role bodies are supported so far"
            )


def _module_text(program, collection):
    """Assemble the rolled IR: one role func per predicate tag and one spmw.map per mapped unit."""
    role_funcs = []
    map_ops = []
    operands = _tensor_operands(program)
    if not operands:
        # the map's assembly format needs at least one tensor operand; a topology-only region
        # (no memref args, e.g. the dialect tests) keeps a benign placeholder.
        operands = [("%arg0", "memref<1xf32>")]
    operand_ssa = ", ".join(ssa for ssa, _ in operands)
    operand_types = ", ".join(ty for _, ty in operands)
    top_params = ", ".join(f"{ssa}: {ty}" for ssa, ty in operands)
    for decl in collection.maps:
        _check_role_bodies_transcribable(decl)
        role_attrs = []
        for name, missing, predicate, body in _roles_of(decl):
            sym = f"{program.name}_{decl.unit.name}_{name}"
            func = None
            ports_text = ""
            if not missing:
                func = _interior_role_func(program, decl, sym, body)
                if func is not None:
                    # the transcribed compute func streams over every port, in the sorted
                    # port-name order it lists them -- declare that as the role's stream ABI
                    ports = ", ".join(f'"{p}"' for p in _interior_ports())
                    ports_text = f", ports = [{ports}]"
            predicate_text = ""
            if predicate is not None:
                # a variant needs an explicit (possibly empty) ports list before its predicate so
                # the RoleAttr parser can tell the two optional trailing fields apart
                ports_text = ports_text or ", ports = []"
                predicate_text = f", predicate = {_predicate_map_text(decl, predicate)}"
            role_funcs.append(func or f"  func.func @{sym}() {{\n    return\n  }}")
            missing_text = ", ".join(f'"{edge}"' for edge in missing)
            role_attrs.append(
                f"#spmw.role<unit = @{sym}, missing = [{missing_text}]"
                f"{ports_text}{predicate_text}>"
            )
        # auto-halo loader/drain datapaths are boundary tasks around the grid, emitted as sibling
        # funcs (not compute-grid roles) and referenced from the map's halo list so spmw-unroll wires
        # them to the edge channels
        for func_text in _halo_role_funcs(program, decl, collection):
            role_funcs.append(func_text)
        roles_text = "[\n      " + ",\n      ".join(role_attrs) + "\n    ]"
        map_attrs = []
        if decl.shard:
            shard_dims = ", ".join(str(s) for s in decl.shard)
            map_attrs.append(f"spmw.shard = array<i64: {shard_dims}>")
        if decl.fold is not None:
            map_attrs.append(
                "fold = array<i64: " + ", ".join(str(f) for f in decl.fold) + ">"
            )
        if decl.unroll is not None:
            map_attrs.append(
                "unroll = array<i64: " + ", ".join(str(u) for u in decl.unroll) + ">"
            )
        halo_attrs = _halo_tasks(program, decl, collection)
        if halo_attrs:
            map_attrs.append(
                "halo = [\n        " + ",\n        ".join(halo_attrs) + "\n      ]"
            )
        attrs_text = "\n      {" + ", ".join(map_attrs) + "}" if map_attrs else ""
        map_ops.append(
            f"    spmw.map ({operand_ssa})\n"
            f"      topology = {_topology_text(decl)}\n"
            f"      roles = {roles_text}"
            f"{attrs_text}\n"
            f"      : {operand_types}"
        )
    func_attrs = []
    channel_attrs = _channel_attrs(program, collection)
    if channel_attrs:
        func_attrs.append(
            "spmw.channels = [\n      " + ",\n      ".join(channel_attrs) + "\n    ]"
        )
    memory_attrs = _memory_attrs(program, collection)
    if memory_attrs:
        func_attrs.append(
            "spmw.memory = [\n      " + ",\n      ".join(memory_attrs) + "\n    ]"
        )
    bank_attrs = _bank_function_attrs(program, collection)
    if bank_attrs:
        func_attrs.append(
            "spmw.bank_functions = [\n      " + ",\n      ".join(bank_attrs) + "\n    ]"
        )
    double_attrs = _double_buffer_attrs(program, collection)
    if double_attrs:
        func_attrs.append("spmw.double_buffers = [" + ", ".join(double_attrs) + "]")
    attrs_text = " attributes {" + ", ".join(func_attrs) + "}" if func_attrs else ""
    top = (
        f"  func.func @{program.name}({top_params}){attrs_text} {{\n"
        + "\n".join(map_ops)
        + "\n    return\n  }"
    )
    return "module {\n" + "\n".join(role_funcs + [top]) + "\n}"


def _parse_module(text):
    # Imported lazily so the surface does not require the compiled backend to be present.
    # pylint: disable=import-outside-toplevel,no-name-in-module
    from ._mlir.ir import Context, Location, Module
    from ._mlir.dialects import allo as allo_d

    context = Context()
    allo_d.register_dialect(context)
    with context, Location.unknown():
        return Module.parse(text)


def lower(program):
    """Lower an SPMW region to the rolled ``spmw.map`` IR and return the parsed module.

    The module carries one ``spmw.map`` op per mapped unit (with a typed ``#spmw.topology`` and
    one ``#spmw.role`` per predicate tag) and one ``func.func`` per role — never ``P0*P1``
    per-grid-point functions. Role bodies are signature-only here; compiling the work-unit
    datapath into them is the next step.
    """
    if not isinstance(program, Region):
        raise SPMWError("spmw.lower expects an @spmw.region")
    collection = _validate_collection(_collect(program))
    return _parse_module(_module_text(program, collection))


def _role_specs(decl):
    """The predicate tags of a mapped unit as ``(name, missing_port_set)`` pairs."""
    specs = [("interior", frozenset())]
    for edges, _ in decl.unit.roles:
        specs.append(("_".join(edges), frozenset(edges)))
    return specs


def _assign_role(boundary, specs):
    """Pick the role for a grid point given the set of links missing at its boundary.

    A role applies where all of its declared missing links are actually missing, so the
    candidates are the roles whose missing set is a subset of the boundary; the most specific
    (largest missing set) wins. Interior (empty missing set) is the always-available fallback;
    two incomparable roles that both fit are a genuine ambiguity and are rejected.
    """
    fits = [(name, missing) for name, missing in specs if missing <= boundary]
    best_size = max(len(missing) for _, missing in fits)
    best = [name for name, missing in fits if len(missing) == best_size]
    if len(best) > 1:
        raise SPMWError(
            f"ambiguous role for boundary {sorted(boundary)}: {best}; declare a role "
            f"for that exact boundary"
        )
    return best[0]


def _check_role_assignment(program, collection):
    """Raise :class:`SPMWError` if any grid point has no assignable, or an ambiguous, role.

    Role selection is performed structurally by the ``spmw-unroll`` pass; this front-end check runs
    the same subset rule (:func:`_assign_role`) over the grid first so an ambiguous or unassignable
    boundary is reported as an ``SPMWError`` before lowering rather than as an MLIR pass failure.
    """
    del program  # signature kept parallel to the other lowering helpers
    for decl in collection.maps:
        specs = _role_specs(decl)
        for coord in decl.topology.coords():
            _assign_role(decl.topology.boundary_ports_at(coord), specs)


def _run_module_pass(module, pipeline):
    """Run an MLIR pass ``pipeline`` in place on a parsed ``module``."""
    # pylint: disable=import-outside-toplevel,no-name-in-module
    from ._mlir.passmanager import PassManager

    with module.context:
        PassManager.parse(f"builtin.module({pipeline})").run(module.operation)


def unroll(program):
    """Expand the rolled map into the per-PID simulator form and return the parsed module.

    Lowers the region to the rolled ``spmw.map`` and then runs the ``spmw-unroll`` MLIR pass, which
    replaces the map with one ``func.call`` per grid point to the role assigned by the links missing
    at that point's boundary. The call count is ``O(P0*P1)`` — the simulator form — but the role
    ``func.func`` count stays ``O(#roles)``: the bodies are never cloned per grid point.
    """
    if not isinstance(program, Region):
        raise SPMWError("spmw.unroll expects an @spmw.region")
    collection = _validate_collection(_collect(program))
    _check_role_assignment(program, collection)
    module = _parse_module(_module_text(program, collection))
    _run_module_pass(module, "spmw-unroll")
    return module


# Targets served by desugaring a recognized region to allo.dataflow (simulator + the HLS toolchain).
_DATAFLOW_TARGETS = frozenset(
    {"simulator", "vitis_hls", "vivado_hls", "tapa", "ihls", "catapult"}
)


def build(program, target="simulator", **kwargs):
    """Validate and build an SPMW program for a target.

    ``target="ir"`` returns the rolled ``spmw.map`` module and ``target="unroll"`` the per-PID
    module. ``target="rolled"`` emits the rolled O(#roles) systolic top as a self-contained Vitis
    HLS project (``kernel.cpp`` + ``run.tcl``). ``target="simulator"`` and the HLS targets
    (``"vitis_hls"`` etc., with ``mode=``) desugar a recognized systolic-mesh region to
    ``allo.dataflow`` and build it through the existing backend.
    """
    validate(program)
    if target == "ir":
        return lower(program)
    if target == "unroll":
        return unroll(program)
    if target == "rolled":
        # pylint: disable=import-outside-toplevel
        from .spmw_hls import _DEFAULT_FREQUENCY_MHZ, _DEFAULT_PART, emit_rolled_project

        return emit_rolled_project(
            program,
            project=kwargs.get("project"),
            part=kwargs.get("part", _DEFAULT_PART),
            frequency=kwargs.get("frequency", _DEFAULT_FREQUENCY_MHZ),
            testbench=kwargs.get("testbench", False),
        )
    if target == "rtl":
        # pylint: disable=import-outside-toplevel
        from .spmw_rtl import emit_structural_verilog

        return emit_structural_verilog(program)
    if target == "vitis_rtl":
        # pylint: disable=import-outside-toplevel
        from .spmw_rtl import emit_vitis_rtl_project

        return emit_vitis_rtl_project(program, project=kwargs.get("project"))
    if target == "rolled_simulator":
        # pylint: disable=import-outside-toplevel
        from .spmw_rollsim import build_rolled_simulator

        return build_rolled_simulator(program)
    if target in _DATAFLOW_TARGETS:
        # pylint: disable=import-outside-toplevel
        from .spmw_datapath import build_dataflow

        return build_dataflow(program, target=target, **kwargs)
    raise NotImplementedError(
        f"SPMW code generation for target={target!r} is not yet implemented; use "
        f"target='simulator'/'rolled_simulator'/'vitis_hls'/'ir'/'unroll'/'rolled'/'rtl'/'vitis_rtl'"
    )
