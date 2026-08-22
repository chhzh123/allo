# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Topologies -- one model for all interconnect.

A topology is constructed over an Interface and its ``link`` function gives, per
site, the far end of that site's ports.  Two forms, one mechanism: the
coordinate form for anything neighbour-based, the key form for permutations and
collectives.  ``to(...)`` is ``key(...)`` with the label computed for you, so
both lower to the same channels.
"""

import itertools

from .errors import (
    SPMWOwnershipError,
    SPMWTopologyError,
    SPMWTypeError,
)
from .interface import unwrap
from .ports import STREAM, IN, OUT


class To:
    """A coordinate-form edge: the far end is a port at a named site."""

    __slots__ = ("coords", "port")

    def __init__(self, coords, port):
        self.coords = tuple(coords) if isinstance(coords, (tuple, list)) else (coords,)
        self.port = port

    def __repr__(self):
        return f"to({self.coords}, {self.port})"


class Key:
    """A key-form endpoint: this end's local half of a pairing.

    Keys stay plain hashable labels deliberately -- a key names a channel, and a
    channel's only declaration site *is* the rendezvous of the ports naming it.
    """

    __slots__ = ("label",)

    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f"key{self.label}"

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        return isinstance(other, Key) and self.label == other.label


def to(coords, port):
    """Name the far end of an ``Out`` port as a port at another site."""
    return To(coords, port)


def key(*label):
    """Name this end's half of a channel by a shared label."""
    return Key(label if len(label) != 1 else label[0])


class Channel:
    """One wire: a single writer, its readers, and the declared element type.

    Coordinate and key form both produce these, which is why everything
    downstream has one path rather than two.
    """

    __slots__ = ("cid", "dtype", "shape", "depth", "writer", "readers", "label")

    def __init__(self, cid, dtype, shape, depth, label=None):
        self.cid = cid
        self.dtype = dtype
        self.shape = shape
        self.depth = depth
        self.writer = None  # (site, port) or None when fed by a binding
        self.readers = []  # [(site, port)]
        self.label = label

    def __repr__(self):
        return f"<channel {self.cid} w={self.writer} r={self.readers}>"


class Topology:
    """A grid plus a link rule, typed by an Interface."""

    def __init__(self, iface, grid, link=None, name=None):
        self.iface = iface
        self.base = unwrap(iface)
        self.grid = tuple(int(g) for g in grid)
        self.link = link
        self.name = name or f"topology{self.grid}"
        if any(g <= 0 for g in self.grid):
            raise SPMWTopologyError(f"grid {self.grid} has a non-positive extent")
        self.channels = {}
        self._bound = {}
        self._attach = {}
        self._elaborate()

    # -- construction ------------------------------------------------------

    def sites(self):
        return itertools.product(*(range(g) for g in self.grid))

    def in_grid(self, coords):
        return len(coords) == len(self.grid) and all(
            0 <= c < g for c, g in zip(coords, self.grid)
        )

    def _elaborate(self):
        """Evaluate the link rule at every site and build the channel set."""
        raw = {}
        for site in self.sites():
            entries = {} if self.link is None else self.link(*site)
            if entries is None:
                entries = {}
            if not isinstance(entries, dict):
                raise SPMWTopologyError(
                    f"`{self.name}`'s link rule returned {type(entries).__name__} at "
                    f"site {site}; it must return a dict keyed by port symbols."
                )
            for port, target in entries.items():
                self._check_owned(port, site)
                raw[(site, port)] = target

        coord_entries = {k: v for k, v in raw.items() if isinstance(v, To)}
        key_entries = {k: v for k, v in raw.items() if isinstance(v, Key)}
        unknown = set(raw) - set(coord_entries) - set(key_entries)
        if unknown:
            site, port = next(iter(unknown))
            raise SPMWTopologyError(
                f"`{self.name}` maps `{port}` at site {site} to "
                f"{raw[(site, port)]!r}; expected spmw.to(...) or spmw.key(...)."
            )

        self._build_coordinate(coord_entries)
        self._build_key(key_entries)
        self._compute_bound()

    def _check_owned(self, port, site):
        if not self.base.owns(port):
            owner = getattr(port, "owner", None)
            owner_name = owner.__name__ if owner is not None else "?"
            raise SPMWOwnershipError(
                f"`{self.name}`'s link rule at site {site} names `{port}`, which "
                f"belongs to `{owner_name}`, not to this topology's interface "
                f"`{self.base.__name__}`."
            )

    def _build_coordinate(self, entries):
        """Declare each edge once, from its source; derive the In side by inversion."""
        for (site, port), target in entries.items():
            if not (port.protocol == STREAM and port.direction == OUT):
                raise SPMWTopologyError(
                    f"`{self.name}` gives `{port}` a coordinate target at site "
                    f"{site}, but only Out ports appear in a coordinate link "
                    f"rule; the In side is derived by inversion. `{port}` is "
                    f"{port.type_str()}."
                )
            self._check_owned(target.port, site)
            far = target.port
            if not (far.protocol == STREAM and far.direction == IN):
                raise SPMWTopologyError(
                    f"`{self.name}`: `{port}` at site {site} targets `{far}`, "
                    f"which is {far.type_str()}. A coordinate edge must land on "
                    f"an In port."
                )
            if far.dtype != port.dtype or far.shape != port.shape:
                raise SPMWTypeError(
                    f"`{self.name}`: `{port}` is {port.type_str()} but its far end "
                    f"`{far}` is {far.type_str()}; element types must agree."
                )
            if not self.in_grid(target.coords):
                # Off-grid far end: the Out is unbound, its put elided.
                continue
            cid = (target.coords, far)
            chan = self.channels.get(cid)
            if chan is None:
                chan = Channel(
                    cid, far.dtype, far.shape, max(port.depth, far.depth), label=None
                )
                self.channels[cid] = chan
            if chan.writer is not None:
                raise SPMWTopologyError(
                    f"`{self.name}`: `{far}` at site {target.coords} is written by "
                    f"both {chan.writer} and ({site}, {port}). A stream has one "
                    f"writer; there is no implicit arbitration."
                )
            chan.writer = (site, port)
            chan.readers.append((target.coords, far))

    def _build_key(self, entries):
        """Both ends name a shared label and rendezvous on it."""
        groups = {}
        for (site, port), label in entries.items():
            if port.protocol != STREAM:
                raise SPMWTopologyError(
                    f"`{self.name}`: `{port}` at site {site} is given a key, but it "
                    f"is {port.type_str()}. Keys name stream channels."
                )
            groups.setdefault(label, []).append((site, port))
        for label, ends in groups.items():
            writers = [(s, p) for s, p in ends if p.direction == OUT]
            readers = [(s, p) for s, p in ends if p.direction == IN]
            if len(writers) > 1:
                raise SPMWTopologyError(
                    f"`{self.name}`: key {label!r} has {len(writers)} writers "
                    f"({', '.join(str(p) for _, p in writers)}). One writer with "
                    f"many readers is a broadcast; many writers is an error."
                )
            dtypes = {(p.dtype, p.shape) for _, p in ends}
            if len(dtypes) > 1:
                raise SPMWTypeError(
                    f"`{self.name}`: key {label!r} is named by ports of differing "
                    f"types: {', '.join(p.type_str() for _, p in ends)}."
                )
            sample = ends[0][1]
            cid = ("key", label)
            chan = Channel(
                cid,
                sample.dtype,
                sample.shape,
                max(p.depth for _, p in ends),
                label=label,
            )
            chan.writer = writers[0] if writers else None
            chan.readers = readers
            self.channels[cid] = chan

    def _compute_bound(self):
        """Per site, the set of ports the topology binds.

        An ``Out`` is bound when its far end exists; an ``In`` when it has a
        producer.  Memory ports are never bound here -- a fabric binding covers
        them, checked at elaboration.
        """
        bound = {site: set() for site in self.sites()}
        attach = {}
        for chan in self.channels.values():
            # An Out is bound when something reads it; an In when something
            # writes it. A key with no reader is an array output, not a wire.
            if chan.writer is not None:
                site, port = chan.writer
                attach[(site, port)] = chan
                if chan.readers:
                    bound[site].add(port)
            for site, port in chan.readers:
                attach[(site, port)] = chan
                if chan.writer is not None:
                    bound[site].add(port)
        for site in bound:
            for sym in self.base.ports():
                if sym.protocol != STREAM:
                    bound[site].add(sym)
        self._bound = {site: frozenset(ports) for site, ports in bound.items()}
        self._attach = attach

    # -- queries -----------------------------------------------------------

    def signature(self, site):
        """The set of ports bound at ``site``."""
        return self._bound[site]

    def signatures(self):
        """Every distinct site signature, and the sites carrying it.

        A 2-D mesh yields nine -- interior, four edges, four corners -- at any
        size, which is the property the whole compilation model rests on.
        """
        classes = {}
        for site, sig in self._bound.items():
            classes.setdefault(sig, []).append(site)
        return classes

    def unbound_sites(self, port):
        """Sites where ``port`` has no counterpart, wherever they occur."""
        return [site for site, sig in self._bound.items() if port not in sig]

    def channel_for(self, site, port):
        """The channel a site's port is attached to, or None when it has none.

        Attached is not the same as bound: a key with a writer and no readers
        attaches its writer but leaves the port unbound.
        """
        return self._attach.get((site, port))

    def __repr__(self):
        return f"<Topology {self.name} {self.base.__name__} grid={self.grid}>"


def Grid(shape, iface=None):  # pylint: disable=invalid-name
    """The linkless topology: pure replication, every port a boundary."""
    shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
    if iface is None:
        return _PendingGrid(shape)
    return Topology(iface, shape, link=None, name=f"Grid{shape}")


class _PendingGrid:
    """``Grid(shape)`` before its interface is known; resolved at ``place``."""

    __slots__ = ("shape",)

    def __init__(self, shape):
        self.shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)

    def resolve(self, iface):
        return Grid(self.shape, iface)

    def __repr__(self):
        return f"Grid({self.shape})"


def mesh(iface, shape):
    """The 4-neighbour mesh, written once against any NSEW-shaped contract."""
    base = unwrap(iface)
    for name in ("east", "west", "north", "south"):
        if name not in base.__ports__:
            raise SPMWTopologyError(
                f"spmw.mesh needs `{name}` on `{base.__name__}`; it declares "
                f"{', '.join(base.__ports__) or '(nothing)'}."
            )
    east, west = base.__ports__["east"], base.__ports__["west"]
    south, north = base.__ports__["south"], base.__ports__["north"]
    return Topology(
        iface,
        shape,
        link=lambda i, j: {
            east: to((i, j + 1), west),
            south: to((i + 1, j), north),
        },
        name=f"mesh{tuple(shape)}",
    )


def ring(iface, size, forward=None, backward=None):
    """A 1-D ring; ``forward``/``backward`` name the two ports."""
    base = unwrap(iface)
    fwd = forward if forward is not None else base.__ports__["out"]
    bwd = backward if backward is not None else base.__ports__["in"]
    return Topology(
        iface,
        (size,),
        link=lambda i: {fwd: to(((i + 1) % size,), bwd)},
        name=f"ring({size})",
    )
