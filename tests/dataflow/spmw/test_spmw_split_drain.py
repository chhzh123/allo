# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The matched GEMM again, with the drain taken out of the PE.

`test_spmw_autosa_match.py` fuses the two: its PE computes a MAC and *then*
forwards `row` results arriving from above, so every PE carries a counter, a
comparison against a runtime coordinate, and a variable-latency FSM alongside
its DSP. Measured at 16x16 that costs a great deal:

| | SPMW fused | AutoSA |
|---|---|---|
| PE array LUT | 25,293 | 9,310 |
| CARRY8 in the PEs | 1,440 | ~0 |

AutoSA's PE is a pure multiply-accumulate and its drain is a separate
`C_drain_IO_L1_out` network. This is that split, expressed the way SPMW expresses
anything structural -- as a second `place()` on the same grid, joined to the
first with `link`.

The arithmetic is unchanged, the chained distribution is unchanged, and the
result arrives in the same order. Only the boundary between the two units moves,
which is exactly the kind of change the design doc claims should be cheap to
make.
"""

import re

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int8, int32

SIZE = 4


def split_drain_of(size, specialise=False):
    """The matched design with the drain as its own placement.

    With ``specialise`` the drain's row becomes part of its *role* rather than
    an input to it, so `for _i in range(row)` gets a constant trip count.
    """
    n = size

    class MacIO(spmw.Interface):
        west = spmw.In(int8)
        north = spmw.In(int8)
        east = spmw.Out(int8)
        south = spmw.Out(int8)
        # Its own result, and nothing else: the PE no longer carries anyone
        # else's on its way past.
        c = spmw.Out(int32)

    class DrainIO(spmw.Interface):
        """One link of the drain chain: emit the PE's result here, then pass on
        what the drain above has already collected."""

        mine = spmw.In(int32)
        up = spmw.In(int32)
        down = spmw.Out(int32)

    class FeedIO(spmw.Interface):
        up = spmw.In(int8[n])
        down = spmw.Out(int8[n])
        lane = spmw.Out(int8)

    mesh = spmw.Topology(
        MacIO,
        grid=(n, n),
        link=lambda i, j: {
            MacIO.east: spmw.to((i, j + 1), MacIO.west),
            MacIO.south: spmw.to((i + 1, j), MacIO.north),
        },
    )
    drains = spmw.Topology(
        DrainIO,
        grid=(n, n),
        link=lambda i, j: {DrainIO.down: spmw.to((i + 1, j), DrainIO.up)},
    )
    chain = spmw.Topology(
        FeedIO,
        grid=(n,),
        link=lambda i: {FeedIO.down: spmw.to((i + 1,), FeedIO.up)},
    )

    @spmw.unit
    def pe(io: MacIO):
        # No `site`: a pure MAC with a fixed trip count, the same at every
        # position. This is the whole point of the split.
        acc: int32 = 0
        for k in range(n):
            a = io.west.get()
            b = io.north.get()
            acc += a * b
            io.east.put(a)
            io.south.put(b)
        io.c.put(acc)

    @spmw.unit
    def drain(io: DrainIO, site: spmw.Site):
        row, _col = site.rank
        io.down.put(io.mine.get())
        for _i in range(row):
            io.down.put(io.up.get())

    @spmw.unit
    def feed(io: FeedIO, site: spmw.Site):
        (slot,) = site.rank
        for k in range(n):
            packed: int8[n] = io.up.get()
            io.lane.put(packed[slot])
            io.down.put(packed)

    @spmw.fabric
    def g(At: int8[n, n], Bt: int8[n, n], Ct: int32[n, n]):
        P = spmw.place(pe, on=mesh)
        D = spmw.place(drain, on=drains, specialise=(0,) if specialise else ())
        Fa = spmw.place(feed, on=chain)
        Fb = spmw.place(feed, on=chain)
        spmw.stream_in(At, into=Fa.up, index=(...,))
        spmw.stream_in(Bt, into=Fb.up, index=(...,))
        spmw.link(Fa.lane, to=P.west)
        spmw.link(Fb.lane, to=P.north)
        # The only new wire: each PE hands its result to the drain beside it.
        spmw.link(P.c, to=D.mine)
        # The chain starts empty at the top row. Row 0 forwards nothing, so it
        # never reads this, but the port is in its body and has to be covered.
        spmw.stream_in(0, into=D.up)
        spmw.gather(Ct, from_=D.down, index=(D.cols, ...))

    g.spmw_parts = (MacIO, DrainIO, FeedIO, mesh, drains, chain, pe, drain, feed)
    return g


split_drain = split_drain_of(SIZE)


def _operands(size=SIZE, seed=9):
    """`At[k]` is column k of A; `Bt[k]` is row k of B — one token per step."""
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 4, size=(size, size)).astype(np.int8)
    b = rng.integers(-4, 4, size=(size, size)).astype(np.int8)
    return a, b, np.ascontiguousarray(a.T), b, np.zeros((size, size), dtype=np.int32)


def _unpack(ct):
    """Column c's t-th arrival, bottom row first — the same order as fused."""
    return ct[:, ::-1].T


def test_reference_matches_numpy():
    a, b, at, bt, ct = _operands()
    spmw.build(split_drain, target="ref")(at, bt, ct)
    np.testing.assert_array_equal(_unpack(ct), a.astype(np.int32) @ b.astype(np.int32))


def test_simulator_matches_numpy():
    a, b, at, bt, ct = _operands()
    spmw.build(split_drain, target="simulator")(at, bt, ct)
    np.testing.assert_array_equal(_unpack(ct), a.astype(np.int32) @ b.astype(np.int32))


def test_the_pe_no_longer_reads_its_position():
    """The PE is coordinate-free; only the drain knows where it is.

    A body that reads `site.rank` needs the fabric to hand it its position on a
    constant-driven stream, and a runtime trip count built on that is what put a
    counter and a comparator in all 256 PEs.
    """
    from allo.spmw import rtl
    from allo.spmw.role_ip import UnitEmitter

    graph = spmw.elaborate(split_drain)
    struct = rtl.StructuralEmitter(graph)
    units = UnitEmitter(graph)
    mesh = next(p for p in struct.placements() if str(p.name).startswith("pe"))
    drains = next(p for p in struct.placements() if str(p.name).startswith("drain"))

    # `coord_ports` is what makes the fabric drive a `_pid` stream into a role,
    # so it is the thing that decides whether the position is really an input.
    for order in range(len(struct.classes(mesh))):
        assert not struct.coord_ports(
            mesh, order
        ), f"pe role {order} still takes a coordinate"
        assert "_pid" not in units.program(mesh, order)[0]
    # the drain does, which is where the runtime trip count now lives
    assert any(
        struct.coord_ports(drains, order)
        for order in range(len(struct.classes(drains)))
    )


def test_it_is_four_placements_now():
    """Mesh, drain, and the two feed chains."""
    from allo.spmw import rtl

    emitter = rtl.StructuralEmitter(spmw.elaborate(split_drain))
    assert len(emitter.placements()) == 4


def test_the_drain_still_reaches_dram_through_one_port_per_column():
    """Splitting must not change the memory interface."""
    from allo.spmw import rtl

    emitter = rtl.StructuralEmitter(spmw.elaborate(split_drain_of(4)))
    assert emitter.movers.masters() == 6  # two chain heads, four drain columns
    emitter.fabric(memory=True)


@pytest.mark.parametrize("size", [2, 3, 8])
def test_it_scales(size):
    a, b, at, bt, ct = _operands(size=size)
    spmw.build(split_drain_of(size), target="ref")(at, bt, ct)
    np.testing.assert_array_equal(_unpack(ct), a.astype(np.int32) @ b.astype(np.int32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_specialising_the_row_makes_the_trip_count_constant():
    """The drain's `range(row)` becomes `range(3)` when the row is its role.

    A role stands for every site with the same wiring and is *told* where it is,
    which keeps the role count independent of the grid -- and makes anything
    derived from the position runtime logic. Measured on this very unit, that
    costs 70 FF and 227 LUT against 11 and 174 for the constant-bounded version.
    """
    from allo.spmw import rtl
    from allo.spmw.role_ip import UnitEmitter

    graph = spmw.elaborate(split_drain_of(4, specialise=True))
    struct = rtl.StructuralEmitter(graph)
    units = UnitEmitter(graph)
    drains = next(p for p in struct.placements() if str(p.name).startswith("drain"))

    sources = [
        units.program(drains, order)[0] for order in range(len(struct.classes(drains)))
    ]
    # one role per row, and none of them reads a coordinate
    assert len(sources) == 4, len(sources)
    for order, text in enumerate(sources):
        assert "_pid0" not in text, text
        assert not [p for p in struct.coord_ports(drains, order) if p.axis == 0]
    # the four bodies differ only in where they think they are: the row is a
    # literal in the tuple `site.rank` unpacks to, which is what makes
    # `range(row)` a constant trip count by the time it reaches HLS
    rows = sorted(
        int(m) for text in sources for m in re.findall(r"row, _col = \((\d+),", text)
    )
    assert rows == [0, 1, 2, 3], rows


def test_specialising_does_not_change_the_numbers():
    a, b, at, bt, ct = _operands()
    spmw.build(split_drain_of(SIZE, specialise=True), target="simulator")(at, bt, ct)
    np.testing.assert_array_equal(_unpack(ct), a.astype(np.int32) @ b.astype(np.int32))


def test_specialising_is_a_role_count_tradeoff():
    """One role per row for the drain, and the mesh untouched."""
    from allo.spmw import rtl

    plain = rtl.cost(spmw.elaborate(split_drain_of(8)))
    special = rtl.cost(spmw.elaborate(split_drain_of(8, specialise=True)))
    assert special["instances"] == plain["instances"]
    assert special["roles"] > plain["roles"]
