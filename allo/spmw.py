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

__all__ = [
    "SPMWError",
    "Topology",
    "Grid",
    "mesh",
    "ring",
    "unit",
    "region",
    "map",
    "stream_in",
    "stream_out",
    "PortContext",
    "PortHandle",
    "validate",
    "role_partition",
    "role_count",
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

    def validate(self):
        """Run every static check; raise :class:`SPMWError` on the first violation."""
        self._validate_grid_rank()
        self._validate_peer_symmetry()
        self._validate_key_channels()
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

    def _validate_key_channels(self):
        sources = {}
        sinks = {}
        for coord in self.coords():
            for port, target in self.links_at(coord).items():
                if not _is_key_form(target):
                    continue
                key, direction = target
                bucket = sources if direction == _SRC else sinks
                bucket.setdefault(key, []).append((coord, port))
        for key in set(sources) | set(sinks):
            n_src, n_sink = len(sources.get(key, [])), len(sinks.get(key, []))
            if n_src != 1 or n_sink != 1:
                raise SPMWError(
                    f"channel key {key!r} must have exactly one source and one sink; "
                    f"got {n_src} source(s) and {n_sink} sink(s) (fan-out/fan-in is a "
                    f"collective extension)"
                )

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


def region():
    """Decorator factory marking a function as an SPMW composition scope."""

    def decorator(fn):
        return Region(fn)

    return decorator


# Every channel defaults to this FIFO depth unless a ``depths={...}`` override says otherwise.
DEFAULT_DEPTH = 2


class _MapDecl:
    """A recorded ``spmw.map`` call: a unit placed over a topology."""

    def __init__(self, work_unit, topology, depths):
        self.unit = work_unit
        self.topology = topology
        self.depths = dict(depths) if depths else {}
        # Resolved per-port FIFO depth: the default for every declared port, overridden per the
        # user's request. This materializes the default so callers do not read the raw override.
        self.port_depths = {port: DEFAULT_DEPTH for port in topology.port_names()}
        self.port_depths.update(self.depths)


class _StreamDecl:
    """A recorded auto-halo ``stream_in``/``stream_out`` declaration."""

    def __init__(self, tensor, work_unit, flow, direction, **extra):
        self.tensor = tensor
        self.unit = work_unit
        self.flow = flow
        self.direction = direction
        self.extra = extra


class _RegionCollection:
    """The maps and streams gathered from one pass over a region body."""

    def __init__(self):
        self.maps = []
        self.streams = []


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


def map(work_unit, grid=None, topo=None, depths=None):
    """Replicate a work unit over a grid, wired by a topology."""
    if not isinstance(work_unit, Unit):
        raise SPMWError("spmw.map expects an @spmw.unit as its first argument")
    topology = _resolve_topology(work_unit, grid, topo)
    if depths:
        ports = topology.port_names()
        for port in depths:
            if port not in ports:
                raise SPMWError(
                    f"depths references undeclared port {port!r}; declared ports: "
                    f"{sorted(ports)}"
                )
    decl = _MapDecl(work_unit, topology, depths)
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


def _collect(program):
    global _active_collector
    collection = _RegionCollection()
    args, kwargs = _make_arguments(program.fn)
    previous = _active_collector
    _active_collector = collection
    try:
        program.fn(*args, **kwargs)
    finally:
        _active_collector = previous
    return collection


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


def _validate_collection(collection):
    if not collection.maps:
        raise SPMWError("region declares no spmw.map")
    for decl in collection.maps:
        decl.topology.validate()
        ports = decl.topology.port_names()
        for edges, _ in decl.unit.roles:
            for edge in edges:
                if edge not in ports:
                    raise SPMWError(
                        f"role on unit {decl.unit.name!r} references undeclared port "
                        f"{edge!r}; declared ports: {sorted(ports)}"
                    )
        _check_body_ports(f"unit {decl.unit.name!r}", decl.unit.interior, ports)
        for edges, body in decl.unit.roles:
            _check_body_ports(f"role {edges} on unit {decl.unit.name!r}", body, ports)
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
        if not isinstance(target, Unit):
            # Named-stage / key-form streaming has no lowering path yet; reject it early.
            raise SPMWError(
                f"stream_{stream.direction} target must be an @spmw.unit (into=/from_=); "
                f"got {target!r}"
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
    """The predicate tags for a mapped unit: the interior body plus each declared role."""
    roles = [("interior", [])]
    for edges, _ in decl.unit.roles:
        roles.append(("_".join(edges), list(edges)))
    return roles


def _topology_text(decl):
    topology = decl.topology
    rep = tuple(0 for _ in topology.grid)
    links = []
    for port, target in sorted(topology.links_at(rep).items()):
        if _is_key_form(target):
            raise SPMWError(
                f"port {port!r}: key-form link lowering is not yet implemented"
            )
        offset = _translation_offset(topology, port)
        if offset is None:
            raise SPMWError(
                f"port {port!r}: only affine-translation peer links are lowerable so far"
            )
        _, peer_port = topology._parse_peer(port, target)
        depth = decl.port_depths.get(port, DEFAULT_DEPTH)
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


def _module_text(program, collection):
    """Assemble the rolled IR: one role func per predicate tag and one spmw.map per mapped unit."""
    role_funcs = []
    map_ops = []
    for decl in collection.maps:
        role_attrs = []
        for name, missing in _roles_of(decl):
            sym = f"{program.name}_{decl.unit.name}_{name}"
            role_funcs.append(f"  func.func @{sym}() {{\n    return\n  }}")
            missing_text = ", ".join(f'"{edge}"' for edge in missing)
            role_attrs.append(f"#spmw.role<unit = @{sym}, missing = [{missing_text}]>")
        roles_text = "[\n      " + ",\n      ".join(role_attrs) + "\n    ]"
        map_ops.append(
            f"    spmw.map (%arg0)\n"
            f"      topology = {_topology_text(decl)}\n"
            f"      roles = {roles_text}\n"
            f"      : memref<1xf32>"
        )
    top = (
        f"  func.func @{program.name}(%arg0: memref<1xf32>) {{\n"
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


def _unrolled_text(program, collection):
    """Assemble the per-PID IR: O(#roles) role funcs plus one func.call per grid point."""
    role_funcs = []
    calls = []
    for decl in collection.maps:
        specs = _role_specs(decl)
        for name, _ in specs:
            sym = f"{program.name}_{decl.unit.name}_{name}"
            role_funcs.append(f"  func.func @{sym}() {{\n    return\n  }}")
        for coord in decl.topology.coords():
            role_name = _assign_role(decl.topology.boundary_ports_at(coord), specs)
            sym = f"{program.name}_{decl.unit.name}_{role_name}"
            calls.append(f"    func.call @{sym}() : () -> ()")
    top = f"  func.func @{program.name}() {{\n" + "\n".join(calls) + "\n    return\n  }"
    return "module {\n" + "\n".join(role_funcs + [top]) + "\n}"


def unroll(program):
    """Expand the rolled map into the per-PID simulator form and return the parsed module.

    Every grid point becomes a ``func.call`` to the ``func.func`` of the role it is assigned
    (by which links are missing at its boundary). The call count is ``O(P0*P1)`` — the simulator
    form — but the role ``func.func`` count stays ``O(#roles)``: the bodies are never cloned per
    grid point. Role bodies are signature-only until the datapath lowering lands.
    """
    if not isinstance(program, Region):
        raise SPMWError("spmw.unroll expects an @spmw.region")
    collection = _validate_collection(_collect(program))
    return _parse_module(_unrolled_text(program, collection))


def build(program, target="simulator", **kwargs):
    """Validate and build an SPMW program for a target.

    ``target="ir"`` returns the rolled ``spmw.map`` module and ``target="unroll"`` the per-PID
    module. Execution targets need the work-unit datapath lowering, which is not wired up yet.
    """
    validate(program)
    if target == "ir":
        return lower(program)
    if target == "unroll":
        return unroll(program)
    raise NotImplementedError(
        f"SPMW code generation for target={target!r} is not yet implemented; use "
        f"spmw.lower/spmw.unroll (target='ir'/'unroll') for the rolled/per-PID modules"
    )
