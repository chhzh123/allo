# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Placement -- snapping a component onto a topology.

Legal iff the component's Interface matches the topology's.  The returned handle
exposes the placement's aggregate boundary, and only that: nothing crosses a
placement boundary except through a declared port.  Because that boundary is
itself an interface, a fabric wires placements exactly the way a topology wires
units.
"""

from .component import Fabric, Unit
from .errors import SPMWPlacementError
from .index import Axis
from .interface import matches, unwrap
from .ports import STREAM
from .topology import Topology, _PendingGrid


class Bundle:
    """A port's unbound instances, wherever they occur -- rim or interior.

    A bundle has a shape of its own: the placement axes along which membership
    varies, with a partially-open axis reduced to its dense factor.
    """

    __slots__ = ("port", "placement", "sites", "_shape", "_axes", "_dense")

    def __init__(self, port, placement, sites):
        self.port = port
        self.placement = placement
        self.sites = tuple(sorted(sites))
        self._shape, self._axes, self._dense = _dense_shape(self.sites, placement)

    @property
    def shape(self):
        """The bundle's dense shape, or None when it is not a product of axis subsets."""
        return self._shape

    @property
    def axes(self):
        """The placement axes along which membership varies, as symbols."""
        return self._axes

    @property
    def is_dense(self):
        return self._dense

    def __len__(self):
        return len(self.sites)

    def __iter__(self):
        return iter(self.sites)

    def __repr__(self):
        shape = "x".join(str(s) for s in self._shape) if self._shape else "?"
        return f"<bundle {self.port} n={len(self.sites)} shape={shape}>"


def _dense_shape(sites, placement):
    """Whether an unbound set is a product of per-axis subsets, and its shape.

    ``grouped_mxu``'s ``a_in`` at G groups is R x G -- rows x open columns --
    never R x C, and never a flat list.
    """
    if not sites:
        return (), (), True
    rank = len(placement.grid)
    projections = [sorted({s[a] for s in sites}) for a in range(rank)]
    product = 1
    for proj in projections:
        product *= len(proj)
    dense = product == len(sites)
    shape, axes = [], []
    for a, proj in enumerate(projections):
        if len(proj) == 1 and len(sites) > 1:
            # A degenerate axis is not one the bundle varies along.
            continue
        shape.append(len(proj))
        axes.append(placement.axes[a])
    return tuple(shape), tuple(axes), dense


class MemGrid:
    """The grid of per-site bricks exported by a ``MemOut`` port."""

    __slots__ = ("port", "placement", "sites")

    def __init__(self, port, placement, sites):
        self.port = port
        self.placement = placement
        self.sites = tuple(sorted(sites))

    @property
    def shape(self):
        return self.placement.grid

    @property
    def axes(self):
        return self.placement.axes

    def __iter__(self):
        return iter(self.sites)

    def __len__(self):
        return len(self.sites)

    def __repr__(self):
        return f"<memgrid {self.port} over {self.placement.grid}>"


_AXIS_NAMES = ("rows", "cols", "planes")


class Placement:
    """A component placed on a topology.

    Exposes bundles for unbound stream ports, memory grids for exported
    ``MemOut`` ports, and the grid's axis symbols.
    """

    def __init__(
        self, component, topology, depths=None, fold=None, unroll=None, layout=None
    ):
        self.component = component
        self.topology = topology
        self.grid = topology.grid
        self.depths = dict(depths or {})
        self.fold = dict(fold or {})
        self.unroll = dict(unroll or {})
        self.layout = dict(layout or {})
        self.iface = unwrap(topology.iface)
        self.name = getattr(component, "name", "component")

        self.axes = tuple(
            Axis(
                _AXIS_NAMES[i] if i < len(_AXIS_NAMES) else f"axis{i}",
                extent,
                resolve=_coord_resolver(i),
            )
            for i, extent in enumerate(self.grid)
        )
        self.roles = {}
        self._bundles = {}
        self._memgrids = {}
        self._bind_roles()
        self._build_boundary()

    # -- axis sugar --------------------------------------------------------

    @property
    def rows(self):
        return self.axes[0]

    @property
    def cols(self):
        if len(self.axes) < 2:
            raise SPMWPlacementError(
                f"`{self.name}` is placed on a rank-{len(self.grid)} grid; it has "
                f"no column axis."
            )
        return self.axes[1]

    def env(self, site):
        """Bind this placement's axis symbols to one site's coordinates."""
        return {axis: coord for axis, coord in zip(self.axes, site)}

    # -- roles -------------------------------------------------------------

    def _bind_roles(self):
        """Assign a body to every site from its signature.

        Selection is total here: roles cover sites whose *behaviour* differs, and
        everything else runs the default body.  Whether an unbound ``In`` that
        body reads is actually fed by a binding is checked at build, where the
        set of bindings is known and the diagnostic can name the missing one.
        """
        if not isinstance(self.component, Unit):
            return
        for sig, sites in self.topology.signatures().items():
            body = self.component.body_for(sig)
            for site in sites:
                self.roles[site] = body

    def role_classes(self):
        """Distinct bodies in use, and the sites running each.

        The count is bounded by the role count and constant as the grid scales.
        """
        classes = {}
        for site, body in self.roles.items():
            classes.setdefault(id(body), (body, []))[1].append(site)
        return {body: sites for body, sites in classes.values()}

    # -- boundary ----------------------------------------------------------

    def _build_boundary(self):
        for sym in self.iface.ports():
            if sym.protocol == STREAM:
                sites = self.topology.unbound_sites(sym)
                if sites:
                    self._bundles[sym] = Bundle(sym, self, sites)
            else:
                # Every memory port is a per-site brick the fabric must bind:
                # MemIn from a shard or a stationary, MemOut into a gather.
                self._memgrids[sym] = MemGrid(sym, self, list(self.topology.sites()))

    def __getattr__(self, name):
        # Bundles and memory grids are reached by the port's own name.
        iface = self.__dict__.get("iface")
        bundles = self.__dict__.get("_bundles", {})
        memgrids = self.__dict__.get("_memgrids", {})
        if iface is not None and name in iface.__ports__:
            sym = iface.__ports__[name]
            if sym in bundles:
                return bundles[sym]
            if sym in memgrids:
                return memgrids[sym]
            raise SPMWPlacementError(
                f"`{self.name}.{name}` has no unbound instances: the link rule "
                f"feeds every one of them, so there is nothing for a fabric to wire."
            )
        raise AttributeError(f"`{type(self).__name__}` has no attribute `{name}`")

    def bundle(self, port):
        """The unbound instances of ``port``, or an empty bundle."""
        return self._bundles.get(port, Bundle(port, self, ()))

    def memgrid(self, port):
        return self._memgrids.get(port)

    def sites(self):
        return self.topology.sites()

    def __repr__(self):
        return f"<placement {self.name} on {self.topology.name}>"


def _coord_resolver(axis_index):
    def resolve(env):
        coords = env.get("__coords__")
        if coords is None:
            raise SPMWPlacementError("axis evaluated without site coordinates")
        return coords[axis_index]

    return resolve


def place(component, on, depths=None, fold=None, unroll=None, layout=None):
    """Instantiate ``component`` on the topology ``on``.

    Legal iff the component's Interface and the topology's match.  The error is
    the Lego error and reads like one.
    """
    if not isinstance(component, (Unit, Fabric)):
        raise SPMWPlacementError(
            f"place expects an @spmw.unit or @spmw.fabric, got "
            f"{type(component).__name__}."
        )
    iface = component.iface if isinstance(component, Unit) else component.io
    if iface is None:
        raise SPMWPlacementError(
            f"fabric `{component.name}` declares no `io=`, so it is not a "
            f"placeable component."
        )
    if isinstance(on, _PendingGrid):
        on = on.resolve(iface)
    if not isinstance(on, Topology):
        raise SPMWPlacementError(
            f"place(on=...) expects a Topology or Grid, got {type(on).__name__}."
        )
    if not matches(iface, on.iface):
        _report_mismatch(component, unwrap(iface), unwrap(on.iface), on)
    placement = Placement(
        component, on, depths=depths, fold=fold, unroll=unroll, layout=layout
    )
    from .graph import register_placement  # pylint: disable=import-outside-toplevel

    register_placement(placement)
    return placement


def _report_mismatch(component, comp_iface, topo_iface, topology):
    """Name the socket that does not exist, not an anonymous structural diff."""
    comp_ports = comp_iface.__ports__
    topo_ports = topo_iface.__ports__
    for name, sym in comp_ports.items():
        if name not in topo_ports:
            raise SPMWPlacementError(
                f"`{component.name}` declares port `{name}: {sym.type_str()}` but "
                f"topology `{topology.name}` has no such socket."
            )
    for name, sym in comp_ports.items():
        other = topo_ports[name]
        if (sym.protocol, sym.direction, sym.dtype, sym.shape) != (
            other.protocol,
            other.direction,
            other.dtype,
            other.shape,
        ):
            raise SPMWPlacementError(
                f"`{component.name}` declares `{name}: {sym.type_str()}` but "
                f"topology `{topology.name}`'s socket is {other.type_str()}."
            )
    raise SPMWPlacementError(
        f"`{component.name}` is written against interface "
        f"`{comp_iface.__name__}` but topology `{topology.name}` is typed by "
        f"`{topo_iface.__name__}`. They are structurally identical -- use "
        f"spmw.structural(...) to accept that, or share one Interface object."
    )
