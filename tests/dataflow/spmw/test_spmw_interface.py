# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interfaces, ports and components -- the contract layer.

These exercise the frontend only; nothing here builds IR.
"""

import pytest

import allo.spmw as spmw
from allo.ir.types import float32, int8

K = 4


class NSEW(spmw.Interface):
    west = spmw.In(float32)
    north = spmw.In(float32)
    east = spmw.Out(float32)
    south = spmw.Out(float32)


class MacIO(NSEW):
    c = spmw.MemOut(float32)


def test_inheritance_shares_symbols():
    """A subclass inherits the same symbol objects, so identity holds."""
    assert MacIO.west is NSEW.west
    assert MacIO.owns(NSEW.west)
    assert not NSEW.owns(MacIO.c)


def test_port_set_is_closed_and_ordered():
    assert [p.name for p in MacIO] == ["west", "north", "east", "south", "c"]
    assert [p.name for p in MacIO.streams(direction="out")] == ["east", "south"]
    assert [p.name for p in MacIO.memories()] == ["c"]
    assert len(MacIO) == 5


def test_type_is_printed_the_way_it_was_declared():
    assert MacIO.west.type_str() == "In(float32)"
    assert MacIO.c.type_str() == "MemOut(float32)"


def test_element_generic_contract():
    """Interface-valued functions, for contracts parameterised by element type."""

    def nsew(dtype):
        return spmw.interface("NSEW", west=spmw.In(dtype), east=spmw.Out(dtype))

    narrow = nsew(int8)
    assert [p.name for p in narrow] == ["west", "east"]
    assert narrow.west is not NSEW.west  # distinct contracts, distinct symbols


def test_misspelled_port_is_caught_at_declaration():
    with pytest.raises(AttributeError, match="wets"):

        @spmw.unit
        def bad(io: MacIO):
            a = io.wets.get()


def test_direction_errors_fire_at_declaration():
    """A wrong-direction touch is rejected where it is written, not at run time."""
    with pytest.raises(spmw.SPMWError, match="west"):

        @spmw.unit
        def puts_on_an_in(io: MacIO):
            io.west.put(1.0)

    with pytest.raises(spmw.SPMWError, match="memory port"):

        @spmw.unit
        def gets_a_memory(io: MacIO):
            x = io.c.get()


def test_site_parameter_is_optional_and_detected():
    @spmw.unit
    def rankless(io: MacIO):
        io.east.put(io.west.get())

    @spmw.unit
    def ranked(io: MacIO, site: spmw.Site):
        i, j = site.rank
        io.east.put(io.west.get())

    assert not rankless.wants_site
    assert ranked.wants_site


def test_role_may_not_touch_its_declared_unbound_ports():
    @spmw.unit
    def cell(io: MacIO):
        io.east.put(io.west.get())

    @cell.role(unbound=(MacIO.west,))
    def cell_west(io: MacIO, site: spmw.Site):
        io.east.put(0.0)

    assert len(cell.roles) == 1
    assert cell.roles[0].unbound == frozenset({MacIO.west})

    with pytest.raises(spmw.SPMWPlacementError, match="north"):

        @cell.role(unbound=(MacIO.north,))
        def bad(io: MacIO):
            x = io.north.get()


def test_role_selection_by_signature():
    @spmw.unit
    def cell(io: MacIO):
        io.east.put(io.west.get())

    @cell.role(unbound=(MacIO.west,))
    def cell_west(io: MacIO, site: spmw.Site):
        io.east.put(0.0)

    interior = frozenset(MacIO)
    assert cell.body_for(interior) is cell

    without_west = frozenset(p for p in MacIO if p is not MacIO.west)
    assert cell.body_for(without_west).name == "cell_west"

    # An unbound Out never forces a role: its puts are simply elided.
    without_east = frozenset(p for p in MacIO if p is not MacIO.east)
    assert cell.body_for(without_east) is cell


def test_nominal_matching_by_default():
    class Twin(spmw.Interface):
        west = spmw.In(float32)
        north = spmw.In(float32)
        east = spmw.Out(float32)
        south = spmw.Out(float32)

    from allo.spmw.interface import matches

    assert matches(NSEW, NSEW)
    assert not matches(Twin, NSEW)
    assert matches(spmw.structural(Twin), NSEW)


def test_a_diamond_agrees_with_attribute_lookup():
    """`__ports__` and ordinary lookup must name the same symbol."""

    class Base(spmw.Interface):
        x = spmw.In(float32)

    class Redeclared(Base):
        x = spmw.In(float32)

    class Extended(Base):
        y = spmw.Out(float32)

    class Both(Extended, Redeclared):
        pass

    assert Both.__ports__["x"] is Both.x
    assert Both.owns(Both.x)
    assert Both.x.owner is Redeclared  # the most derived declaration wins


def test_a_declaration_belongs_to_one_interface():
    """Sharing one declaration would give a single symbol two owners."""
    shared = spmw.In(float32)

    class First(spmw.Interface):
        p = shared

    with pytest.raises(spmw.SPMWError, match="reuses the declaration"):

        class Second(spmw.Interface):
            p = shared

    with pytest.raises(spmw.SPMWError, match="same declaration object"):

        class Twice(spmw.Interface):
            a = b = spmw.In(float32)


def test_rebinding_io_is_refused():
    """The io name is the contract for the whole body."""
    with pytest.raises(spmw.SPMWPlacementError, match="rebinds"):

        @spmw.unit
        def confused(io: MacIO):
            v = io.west.get()
            io = 3
            io.east.put(v)


def test_writing_through_io_is_not_rebinding():
    """`io.c = x` writes a port; only binding the name itself is a rebind."""

    @spmw.unit
    def writes(io: MacIO):
        io.c = io.west.get()

    assert writes.name == "writes"


def test_a_role_on_a_memory_port_is_refused():
    """A memory port is bound by a binding, so it never enters a signature."""

    class WIO(spmw.Interface):
        a_in = spmw.In(float32)
        a_out = spmw.Out(float32)
        w = spmw.MemIn(float32)

    @spmw.unit
    def cell(io: WIO):
        io.a_out.put(io.a_in.get() * io.w)

    with pytest.raises(spmw.SPMWError, match="could never be selected"):

        @cell.role(unbound=(WIO.w,))
        def dead(io: WIO, site: spmw.Site):
            io.a_out.put(0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
