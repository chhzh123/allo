# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolving links into declarable channel arrays.

Coordinate and key links produce the same :class:`~allo.spmw.topology.Channel`
records, so this pass has one input.  What it decides is how each family of
channels is *addressed*: a family whose every edge is the same constant
displacement stays an affine subscript (``fifo[i, j + 1]``), and anything else --
a wrap, a permutation, a broadcast -- falls back to a lookup keyed by site.

Both spellings are exact.  The affine one exists because the flagship shapes are
affine and the generated program should still read like the array it describes.
"""

from .errors import SPMWTopologyError

AFFINE = "affine"
TABLE = "table"


class Family:
    """One declarable array of streams.

    ``kind`` is :data:`AFFINE` -- an array of grid shape, addressed by the
    destination site -- or :data:`TABLE` -- a flat array addressed by a channel
    id looked up per site.
    """

    def __init__(self, name, dtype, block, depth, kind, shape):
        self.name = name
        self.dtype = dtype
        self.block = tuple(block)
        self.depth = depth
        self.kind = kind
        self.shape = tuple(shape)
        self.offset = None  # AFFINE: destination displacement from the writer
        self.slots = {}  # TABLE: (site, port) -> channel id
        # Per port, the bundle whose members this family indexes. When that
        # bundle's sites form a product of arithmetic progressions, a site can
        # compute its own member position instead of looking it up.
        self.geometry = {}
        self.count = 0

    def __repr__(self):
        return f"<family {self.name} {self.kind} {list(self.shape)}>"


class Resolution:
    """Every channel family a placement needs, and how each port addresses one."""

    def __init__(self, placement, prefix):
        self.placement = placement
        self.prefix = prefix
        self.families = {}
        self.port_family = {}

    def family_of(self, port):
        return self.families.get(self.port_family.get(port))

    def __repr__(self):
        return f"<resolution {self.prefix}: {list(self.families)}>"


def resolve(placement, prefix):
    """Group a placement's channels into families and classify their addressing."""
    res = Resolution(placement, prefix)
    topo = placement.topology

    coord, keyed = [], []
    for chan in topo.channels.values():
        (keyed if chan.label is not None else coord).append(chan)

    _resolve_coordinate(res, coord, prefix)
    _resolve_keyed(res, keyed, prefix, placement)
    return res


def _resolve_coordinate(res, channels, prefix):
    """Group by the ordered port pair; keep the affine form when one exists."""
    groups = {}
    for chan in channels:
        if chan.writer is None or not chan.readers:
            continue
        _wsite, wport = chan.writer
        _rsite, rport = chan.readers[0]
        groups.setdefault((wport, rport), []).append(chan)

    for (wport, rport), members in groups.items():
        name = f"{prefix}_{wport.name}_{rport.name}"
        offsets = set()
        for chan in members:
            wsite, _ = chan.writer
            rsite, _ = chan.readers[0]
            offsets.add(tuple(r - w for r, w in zip(rsite, wsite)))
            if len(chan.readers) > 1:
                offsets.add(("fanout",))
        grid = res.placement.grid
        if len(offsets) == 1 and "fanout" not in next(iter(offsets)):
            fam = Family(
                name, wport.dtype, wport.shape, _depth(res, wport, rport), AFFINE, grid
            )
            fam.offset = next(iter(offsets))
        else:
            fam = Family(
                name,
                wport.dtype,
                wport.shape,
                _depth(res, wport, rport),
                TABLE,
                (len(members),),
            )
            for cid, chan in enumerate(sorted(members, key=_chan_key)):
                wsite, _ = chan.writer
                fam.slots[(wsite, wport)] = cid
                for rsite, _rp in chan.readers:
                    fam.slots[(rsite, rport)] = cid
            fam.count = len(members)
        res.families[name] = fam
        res.port_family[wport] = name
        res.port_family[rport] = name


def _resolve_keyed(res, channels, prefix, placement):
    """A key names a channel; the ports naming it are its writer and readers."""
    live = [c for c in channels if c.writer is not None and c.readers]
    if not live:
        return
    ports = set()
    for chan in live:
        ports.add(chan.writer[1])
        for _site, port in chan.readers:
            ports.add(port)
    sample = sorted(ports, key=lambda p: p.name)[0]
    for port in ports:
        if (port.dtype, port.shape) != (sample.dtype, sample.shape):
            raise SPMWTopologyError(
                f"`{placement.name}`'s keyed channels mix element types: "
                f"{port.type_str()} vs {sample.type_str()}."
            )
    name = f"{prefix}_key"
    fam = Family(
        name,
        sample.dtype,
        sample.shape,
        max(p.depth for p in ports),
        TABLE,
        (len(live),),
    )
    for cid, chan in enumerate(sorted(live, key=lambda c: _label_key(c.label))):
        wsite, wport = chan.writer
        fam.slots[(wsite, wport)] = cid
        for rsite, rport in chan.readers:
            fam.slots[(rsite, rport)] = cid
    fam.count = len(live)
    res.families[name] = fam
    for port in ports:
        res.port_family[port] = name


def _depth(res, wport, rport):
    depths = res.placement.depths
    for port in (wport, rport):
        if port in depths:
            return depths[port]
    return max(wport.depth, rport.depth)


def _chan_key(chan):
    return (chan.writer[0], chan.readers[0][0])


def _label_key(label):
    """Order keys deterministically without assuming they are comparable."""
    if isinstance(label, tuple):
        return (0, tuple(_label_key(x) for x in label))
    if isinstance(label, (int, float)):
        return (1, label)
    return (2, str(label))
