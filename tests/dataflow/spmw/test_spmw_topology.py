# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Topologies and placement: link forms, site signatures, bundles, axes."""

import pytest

import allo.spmw as spmw
from allo.ir.types import float32, int8, int32

csample = float32[2]


class NSEW(spmw.Interface):
    west = spmw.In(float32)
    north = spmw.In(float32)
    east = spmw.Out(float32)
    south = spmw.Out(float32)


class MacIO(NSEW):
    c = spmw.MemOut(float32)


@spmw.unit
def pe(io: MacIO):
    acc: float32 = 0
    for k in range(4):
        a = io.west.get()
        b = io.north.get()
        acc += a * b
        io.east.put(a)
        io.south.put(b)
    io.c = acc


# --------------------------------------------------------------------------
# Coordinate form
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [4, 8, 16])
def test_mesh_has_nine_signatures_at_any_size(size):
    """Interior, four edges, four corners -- the count the whole model rests on."""
    topo = spmw.mesh(MacIO, (size, size))
    assert len(topo.signatures()) == 9
    assert sum(len(v) for v in topo.signatures().values()) == size * size


def test_mesh_boundaries():
    topo = spmw.mesh(MacIO, (3, 3))
    assert len(topo.channels) == 12  # 3 rows x 2 east edges + 2 x 3 south edges
    assert topo.unbound_sites(MacIO.west) == [(0, 0), (1, 0), (2, 0)]
    assert topo.unbound_sites(MacIO.north) == [(0, 0), (0, 1), (0, 2)]
    assert topo.unbound_sites(MacIO.east) == [(0, 2), (1, 2), (2, 2)]


def test_bundles_and_axes():
    P = spmw.place(pe, on=spmw.mesh(MacIO, (4, 4)))
    assert P.west.shape == (4,)
    assert P.west.axes == (P.rows,)
    assert P.north.axes == (P.cols,)
    assert P.c.shape == (4, 4)


def test_only_out_ports_appear_in_a_coordinate_rule():
    with pytest.raises(spmw.SPMWTopologyError, match="only Out ports"):
        spmw.Topology(
            MacIO,
            (2, 2),
            link=lambda i, j: {MacIO.west: spmw.to((i, j + 1), MacIO.east)},
        )


def test_a_foreign_port_is_rejected_by_ownership():
    class Other(spmw.Interface):
        east = spmw.Out(float32)
        west = spmw.In(float32)

    with pytest.raises(spmw.SPMWOwnershipError, match="Other"):
        spmw.Topology(
            MacIO,
            (2, 2),
            link=lambda i, j: {Other.east: spmw.to((i, j + 1), MacIO.west)},
        )


def test_element_types_must_agree_across_a_link():
    class Mixed(spmw.Interface):
        a_out = spmw.Out(float32)
        a_in = spmw.In(int8)

    with pytest.raises(spmw.SPMWTypeError, match="element types must agree"):
        spmw.Topology(
            Mixed, (2,), link=lambda i: {Mixed.a_out: spmw.to((i + 1,), Mixed.a_in)}
        )


def test_place_names_the_missing_socket():
    class OtherIO(spmw.Interface):
        psum_in = spmw.In(float32)

    @spmw.unit
    def other(io: OtherIO):
        x = io.psum_in.get()

    with pytest.raises(spmw.SPMWPlacementError, match="no such socket"):
        spmw.place(other, on=spmw.mesh(MacIO, (2, 2)))


# --------------------------------------------------------------------------
# Key form
# --------------------------------------------------------------------------

FFT_N, S = 16, 4
HALF = FFT_N // 2


class BflyIO(spmw.Interface):
    up_in = spmw.In(csample)
    lo_in = spmw.In(csample)
    up_out = spmw.Out(csample)
    lo_out = spmw.Out(csample)


def bfly_pair(s, b):
    span = 1 << s
    up = (b // span) * (2 * span) + (b % span)
    return up, up + span


@spmw.unit
def bfly(io: BflyIO, site: spmw.Site):
    a = io.up_in.get()
    c = io.lo_in.get()
    io.up_out.put(a)
    io.lo_out.put(c)


def test_key_form_pairs_by_rendezvous():
    def links(s, b):
        up, lo = bfly_pair(s, b)
        return {
            BflyIO.up_in: spmw.key(s, up),
            BflyIO.lo_in: spmw.key(s, lo),
            BflyIO.up_out: spmw.key(s + 1, up),
            BflyIO.lo_out: spmw.key(s + 1, lo),
        }

    topo = spmw.Topology(BflyIO, grid=(S, HALF), link=links)
    internal = [c for c in topo.channels.values() if c.writer and c.readers]
    assert len(internal) == (S - 1) * FFT_N

    P = spmw.place(bfly, on=topo)
    assert len(P.up_in) == HALF and len(P.up_out) == HALF
    assert all(site[0] == 0 for site in P.up_in.sites)
    assert all(site[0] == S - 1 for site in P.up_out.sites)


def test_many_writers_on_one_key_is_an_error():
    class KIO(spmw.Interface):
        o = spmw.Out(float32)
        i = spmw.In(float32)

    with pytest.raises(spmw.SPMWTopologyError, match="writers"):
        spmw.Topology(
            KIO, (4,), link=lambda i: {KIO.o: spmw.key("same"), KIO.i: spmw.key("same")}
        )


# --------------------------------------------------------------------------
# Interior boundaries and axis algebra
# --------------------------------------------------------------------------


class WsIO(spmw.Interface):
    a_in = spmw.In(int8)
    a_out = spmw.Out(int8)
    p_in = spmw.In(int32)
    p_out = spmw.Out(int32)
    w = spmw.MemIn(int8)


@spmw.unit
def mac(io: WsIO):
    for m in range(8):
        a = io.a_in.get()
        p = io.p_in.get()
        io.p_out.put(p + a * io.w)
        io.a_out.put(a)


def grouped_mxu(iface, shape, groups):
    """Column slabs; the psum chain serpentines from one slab into the next."""
    rows, cols = shape
    d = cols // groups

    def link(i, j):
        links = {}
        if (j + 1) % d != 0:  # forward activations inside a slab only
            links[iface.a_out] = spmw.to((i, j + 1), iface.a_in)
        if i + 1 < rows:  # psums: down my column, ...
            links[iface.p_out] = spmw.to((i + 1, j), iface.p_in)
        elif j + d < cols:  # ... then on to the next slab's top
            links[iface.p_out] = spmw.to((0, j + d), iface.p_in)
        return links

    return spmw.Topology(iface, shape, link=link, name=f"grouped_mxu(G={groups})")


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_withheld_links_make_interior_boundaries(groups):
    """A link rule that withholds an edge opens a port inside the grid."""
    rows, cols = 16, 16
    d = cols // groups
    P = spmw.place(mac, on=grouped_mxu(WsIO, (rows, cols), groups))

    # a_in is open at every slab west: rows x groups, never rows x cols.
    assert len(P.a_in) == rows * groups
    assert P.a_in.shape == ((rows,) if groups == 1 else (rows, groups))
    assert P.a_in.is_dense
    # The chain is seeded only at slab 0's top and drained only at the last
    # slab's bottom, whatever the grouping.
    assert len(P.p_in) == d
    assert len(P.p_out) == d
    assert all(site == (0, e) for site, e in zip(P.p_in.sites, range(d)))
    assert all(site[0] == rows - 1 and site[1] >= cols - d for site in P.p_out.sites)


def test_split_axes():
    rows, cols, groups = 16, 16, 4
    P = spmw.place(mac, on=grouped_mxu(WsIO, (rows, cols), groups))
    g, e = spmw.split(P.cols, factor=groups)
    assert g.extent == groups
    assert e.extent == cols // groups

    env = {"__coords__": (3, 9)}  # site (k=3, c=9) with d=4
    assert g.eval(env) == 2
    assert e.eval(env) == 1
    assert (g * rows + P.rows).eval(env) == 2 * 16 + 3


def test_split_must_cover_its_axis():
    P = spmw.place(mac, on=grouped_mxu(WsIO, (16, 16), 2))
    with pytest.raises(spmw.SPMWBindingError, match="does not cover"):
        spmw.split(P.cols, factor=5)


def test_linkless_grid():
    class ActIO(spmw.Interface):
        z_in = spmw.In(int32)
        y_out = spmw.Out(int8)

    @spmw.unit
    def act(io: ActIO):
        for m in range(4):
            io.y_out.put(io.z_in.get())

    P = spmw.place(act, on=spmw.Grid((8,)))
    assert P.grid == (8,)
    assert len(P.z_in) == 8 and len(P.y_out) == 8
    (lane,) = P.axes
    assert lane.extent == 8


def test_equally_specific_roles_are_refused():
    """Which of two equal matches runs must not come down to declaration order."""

    @spmw.unit
    def cell(io: MacIO):
        acc: float32 = io.west.get() * io.north.get()
        io.east.put(acc)
        io.south.put(acc)
        io.c = acc

    @cell.role(unbound=(MacIO.west,))
    def only_west(io: MacIO, site: spmw.Site):
        acc: float32 = io.north.get()
        io.east.put(acc)
        io.south.put(acc)
        io.c = acc

    @cell.role(unbound=(MacIO.north,))
    def only_north(io: MacIO, site: spmw.Site):
        acc: float32 = io.west.get()
        io.east.put(acc)
        io.south.put(acc)
        io.c = acc

    # The corner is missing both, and neither role is more specific than the other.
    with pytest.raises(spmw.SPMWPlacementError, match="equally specific"):
        spmw.place(cell, on=spmw.mesh(MacIO, (2, 2)))


def test_bundle_shape_is_comparable_across_sizes():
    """A degenerate axis is not one the bundle varies along, at any size."""
    one = spmw.place(pe, on=spmw.mesh(MacIO, (1, 1)))
    many = spmw.place(pe, on=spmw.mesh(MacIO, (3, 1)))
    assert one.west.shape == () and one.west.axes == ()
    assert many.west.shape == (3,) and many.west.axes == (many.rows,)


def test_a_missing_component_port_is_named():
    """An extra socket on the topology is a dangling wire, and says so."""

    class Small(spmw.Interface):
        a = spmw.In(float32)

    class Big(spmw.Interface):
        a = spmw.In(float32)
        b = spmw.Out(float32)

    @spmw.unit
    def tiny(io: Small):
        x = io.a.get()

    with pytest.raises(spmw.SPMWPlacementError, match="does not declare"):
        spmw.place(tiny, on=spmw.Topology(Big, (2,), link=None))


def test_one_port_may_take_part_in_two_families():
    """A port wired by different links at different sites must still address the
    right array at each of them."""
    from allo.spmw import channels as ch

    class AltIO(spmw.Interface):
        inp = spmw.In(float32)
        a_out = spmw.Out(float32)
        b_out = spmw.Out(float32)

    @spmw.unit
    def alt(io: AltIO):
        v = io.inp.get()
        io.a_out.put(v)
        io.b_out.put(v)

    def links(i):
        return {
            (AltIO.a_out if i % 2 == 0 else AltIO.b_out): spmw.to((i + 1,), AltIO.inp)
        }

    P = spmw.place(alt, on=spmw.Topology(AltIO, (4,), link=links))
    res = ch.resolve(P, "p")
    # `inp` is fed by a different family at even and odd sites.
    assert res.site_family[((1,), AltIO.inp)] != res.site_family[((2,), AltIO.inp)]


def test_family_names_are_injective():
    """`a`/`b_c` and `a_b`/`c` spell the same name; they must not merge."""
    from allo.spmw import channels as ch

    class CollideIO(spmw.Interface):
        a = spmw.Out(float32)
        b_c = spmw.In(float32)
        a_b = spmw.Out(float32)
        c = spmw.In(float32)

    @spmw.unit
    def cell(io: CollideIO):
        io.a.put(io.b_c.get())
        io.a_b.put(io.c.get())

    P = spmw.place(
        cell,
        on=spmw.Topology(
            CollideIO,
            (6,),
            link=lambda i: {
                CollideIO.a: spmw.to((i + 1,), CollideIO.b_c),
                CollideIO.a_b: spmw.to((i + 2,), CollideIO.c),
            },
        ),
    )
    assert len(ch.resolve(P, "p").families) == 2


def test_grid_accepts_a_bare_extent():
    class Small(spmw.Interface):
        a = spmw.In(float32)

    assert spmw.Grid(4, Small).grid == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
