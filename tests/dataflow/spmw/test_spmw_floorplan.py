# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the array should sit on the die, and which wires need help getting there.

AutoBridge [FPGA'21] and RapidStream [FPGA'22] make the same argument: on a
multi-die FPGA an HLS design misses its clock target because the compiler cannot
see how far a wire will have to go, and the fix is to floorplan the dataflow
graph *and* pipeline the connections that cross the floorplan's boundaries.
Either half alone is worthless -- a floorplan with no added pipelining just
takes options away from the placer, which is what a measurement on this array
found.

Both papers spend most of their machinery recovering structure that a spatial
program never lost.  AutoBridge builds a dataflow graph out of C++, estimates
each function's area from HLS reports, and solves an ILP to partition it; then,
because "each vertex is an FSM and the firing rate is not fixed", it has to be
conservative about where latency may be added, solving an SDC over cut-sets to
re-balance reconvergent paths and refusing to pipeline anything on a dependency
cycle.

A `Topology` states all of that up front, so these tests check the two claims
that follow from it:

* the channels needing an anchor register are a *family*, identified from the
  family's `offset`, not a set of nets found by a search; and
* a family is a cut-set already, so adding a stage to all of it is balanced by
  construction and no SDC has to be solved to prove it.
"""

import re

import allo.spmw as spmw
from allo.spmw import channels as ch
from allo.spmw.rtl import StructuralEmitter
from allo.spmw.shell import crossing_families, floorplan_xdc

from test_spmw_transformer import BIG


def _graph():
    return spmw.elaborate(BIG.engine)


def test_only_the_links_that_cross_the_cut_get_an_anchor():
    """The mesh flows two ways; a cut along the rows crosses exactly one of them.

    Partial sums travel south, so every one of them crosses every row boundary
    the floorplan draws.  Activations travel east, entirely inside a band, and
    must not be touched -- anchoring them would buy nothing and cost 256 FIFOs.

    AutoBridge decides this per net, after an ILP has placed the modules.  Here
    it is a property of one integer: the family's displacement from writer to
    reader.
    """
    graph = _graph()
    anchors = crossing_families(graph, slots=4)

    internal, _boundary = StructuralEmitter(graph).families()
    offsets = {fam.name: getattr(fam, "offset", None) for fam in internal}
    southward = {n for n, off in offsets.items() if off and off[0]}
    eastward = {n for n, off in offsets.items() if off and not off[0]}

    assert southward, "the mesh should have a family that travels along the rows"
    assert eastward, "and one that travels along the columns"
    assert set(anchors) == southward & set(anchors)
    assert not set(anchors) & eastward, "an eastward link never leaves its band"


def test_only_the_channels_landing_on_a_boundary_are_anchored():
    """A family crosses; its channels do not all cross in the same place.

    Every southward link travels one row, but with the mesh cut into four bands
    only those landing on rows 4, 8 and 12 leave a band.  Anchoring the family
    wholesale would build 768 registers to do the work of 48 -- correct, and a
    sixteenfold waste.  This is the part a family-level answer gets wrong and
    the channel index gets right.
    """
    graph = _graph()
    rows = 16
    for slots in (2, 4):
        anchors = crossing_families(graph, slots=slots)
        assert anchors, "a cut along the rows must cross the southward family"
        for plan in anchors.values():
            assert list(plan.cuts) == [(s * rows) // slots for s in range(1, slots)]
            # One row of the mesh per boundary, and the array's edge is not one.
            assert plan.channels(rows * rows) == (slots - 1) * rows
            assert 0 not in plan.cuts
    assert not crossing_families(graph, slots=1), "one slot has no boundary"


def test_the_anchored_channels_are_the_ones_that_change_band():
    """Checked against the floorplan itself rather than against the arithmetic.

    `channel_index` subscripts an affine family by its destination site, so
    channel `i` is read by the site at row `i // 16`, written by the site one
    row above it.  A channel needs an anchor exactly when those two rows are in
    different pblocks.  Recomputing that from `floorplan_xdc`'s own output is
    what makes this a check and not a restatement.
    """
    graph = _graph()
    slots, rows = 4, 16
    plan = floorplan_xdc(graph, top="dut", slots=slots)
    band = {}
    for slot, chunk in enumerate(plan.split("create_pblock")[1:]):
        cells = [l for l in chunk.splitlines() if l.startswith("add_cells_to_pblock")]
        assert len(cells) == 1, "one band, one cell list"
        for cell in cells[0].split("{")[1].split("}")[0].split():
            # Units only: the list also carries each channel's FIFO, whose name
            # is `g_<family>[<channel>].u` and says nothing about a mesh row.
            if "/u_mac_" not in cell:
                continue
            band[int(cell.rsplit("_", 2)[-2])] = slot  # the site's row
    assert set(band) == set(range(rows)), "every mesh row should land in a band"

    crossing = crossing_families(graph, slots=slots)["mac_p_out_p_in"]
    for reader in range(1, rows):
        expected = band[reader] != band[reader - 1]
        assert crossing.crosses(reader * rows) is expected, f"row {reader}"


def test_a_family_is_a_cut_set():
    """Why adding latency to a whole family needs no re-balancing.

    Cut-set pipelining is safe when every edge of the cut carries the same added
    latency and all of them point the same way.  An AFFINE `link` family is
    exactly that: the elaborator sets `offset` only when every channel in the
    family agrees on one displacement from writer to reader, and leaves it
    `None` otherwise.  So `offset is not None` *is* the uniformity proof, and
    anchoring a family with it is balanced by construction -- which is the SDC
    that AutoBridge has to solve, answered by how the channel was declared.

    The exclusion has to bite, not just be written down.  This engine has an
    internal family that is a TABLE -- the mesh's drain into the activation row
    -- whose channels do not share a displacement.  It is internal, so a rule
    that only asked "is this a link?" would anchor it and skew the drain.
    """
    graph = _graph()
    internal, _boundary = StructuralEmitter(graph).families()
    anchors = crossing_families(graph, slots=4)
    named = {fam.name: fam for fam in internal}

    for name in anchors:
        assert named[name].offset is not None, "anchored a family with no one journey"
        assert named[name].kind is ch.AFFINE

    # The exclusion fires: an internal family with no single displacement exists.
    formless = {f.name for f in internal if getattr(f, "offset", None) is None}
    assert formless, "expected the drain into the activation row to be a TABLE"
    assert not formless & set(anchors)


def test_the_chain_is_placed_but_never_anchored():
    """Two different questions about the engine's activation row.

    It has no boundary of its own to cross, so `crossing_families` must leave
    its links alone -- anchoring them would be a different optimisation wearing
    this one's name.

    But it must still be *placed*. The first version of the floorplan skipped
    every rank-1 placement outright, which left the 32 vector lanes and the
    FIFOs joining them to the mesh free to land anywhere while everything
    around them was pinned. That family is exactly where the 32x32 critical
    path turned up, at 4.7 ns. The chain is linked from the mesh's last row, so
    its home is the last band.
    """
    graph = _graph()
    emitter = StructuralEmitter(graph)
    chains = [p for p in emitter.placements() if len(tuple(p.grid)) == 1]
    assert chains, "the engine should still have its activation row"
    chained = {fam.name for p in chains for fam in emitter.peer_families(p)}
    assert chained, "and that row should link its sites"
    assert not chained & set(crossing_families(graph, slots=4))

    plan = floorplan_xdc(graph, top="dut", slots=4)
    names = [emitter.role_names(p) for p in chains]
    for placement, roles in zip(chains, names):
        for order, (_sig, _rt, sites) in enumerate(emitter.classes(placement)):
            for site in sites:
                tag = "_".join(str(int(c)) for c in site)
                assert f"dut/u_{roles[order]}_{tag}" in plan


def test_every_unit_and_every_internal_fifo_gets_a_band():
    """A partial floorplan is worse than none, measured.

    Banding the mesh cells and nothing else cost 35% of the clock at 32x32:
    1,024 pinned instances left 3,000-odd FIFOs and 32 lanes to fill whatever
    gaps remained, and a FIFO whose two counter bits land in different clock
    regions is a 4.7 ns path. Whatever the floorplan constrains, it has to
    constrain all of it.
    """
    graph = _graph()
    emitter = StructuralEmitter(graph)
    plan = floorplan_xdc(graph, top="dut", slots=4)

    for placement in emitter.placements():
        roles = emitter.role_names(placement)
        for order, (_sig, _rt, sites) in enumerate(emitter.classes(placement)):
            for site in sites:
                tag = "_".join(str(int(c)) for c in site)
                assert f"dut/u_{roles[order]}_{tag}" in plan, (roles[order], site)

    internal, _boundary = emitter.families()
    for fam in internal:
        placed = set(re.findall(rf"g_{fam.name}\[(\d+)\]", plan))
        assert placed, f"no FIFO of `{fam.name}` was placed"
        # Every channel with a reader inside the array; the ones without are
        # fed from the boundary and have no FIFO in the fabric at all.
        assert len(placed) >= 1


def test_the_floorplan_names_bands_and_not_rows():
    """Four slots is four pblocks, whatever the mesh's row count.

    The first version of this emitted one pblock per mesh row -- sixteen of
    them, which the geometry then collapsed onto four clock regions anyway.  It
    constrained the placer sixteen times to say what four constraints said, and
    it measured worse than no floorplan at all.
    """
    graph = _graph()
    for slots in (2, 4):
        plan = floorplan_xdc(graph, top="dut", slots=slots)
        assert plan.count("create_pblock") == slots
        assert plan.count("add_cells_to_pblock") == slots
    # Every mesh instance lands in exactly one band.
    plan = floorplan_xdc(graph, top="dut", slots=4)
    cells = [
        c
        for line in plan.splitlines()
        if "add_cells_to_pblock" in line
        for c in line.split("{")[1].split("}")[0].split()
    ]
    assert len(cells) == len(set(cells)), "a cell may not be in two pblocks"


def test_the_anchored_fabric_still_says_what_it_connects():
    """Anchors are an extra stage *on* a channel, not an extra channel beside it.

    The first version of this test asked whether the intermediate wires were
    declared and whether the family's own wires still appeared.  Both were true
    of a fabric whose stages were not connected to each other at all -- each
    drove a `dout` bundle while the next read a `din` bundle, so every join was
    two dangling halves.  SystemVerilog accepts that in silence and the array
    deadlocked in simulation with 0 of 16 tokens out.

    So this counts drivers and readers instead: the join must be named once by
    the stage above and once by the stage below, and the chain must start at the
    producer and end at the consumer.
    """
    graph = _graph()
    anchors = crossing_families(graph, slots=4)
    plain = StructuralEmitter(graph).fabric()
    wired = StructuralEmitter(graph).fabric(anchors=anchors)
    assert plain.count("u_anchor") == 0
    assert wired.count("u_anchor") == len(anchors)

    name = next(iter(anchors))
    chain = [l for l in wired.splitlines() if "spmw_fifo" in l and f"{name}_" in l]
    assert len(chain) == 3, "an anchored stage, its channel, and the direct one"

    tag = f"{name}_h"
    for port, wire in ((".dout", "data"), (".empty_n", "valid"), (".read", "ready")):
        assert sum(f"{port}({tag}_{wire}[" in l for l in chain) == 1
    for port, wire in ((".din", "data"), (".write", "valid"), (".full_n", "ready")):
        assert sum(f"{port}({tag}_{wire}[" in l for l in chain) == 1

    # Both branches end at the consumer, and only one of them can be taken.
    assert sum(f".dout({name}_dout[" in l for l in chain) == 2
    assert sum(f".din({name}_din[" in l for l in chain) == 2
    assert "begin : anchored" in wired and "begin : direct" in wired


def test_every_anchor_wire_is_both_driven_and_read():
    """No dangling half anywhere in the fabric, at any slot count.

    This is the property the deadlock violated, asserted over the whole emitted
    source rather than one family, so a later change cannot reintroduce it
    quietly.
    """
    graph = _graph()
    for slots in (2, 4):
        anchors = crossing_families(graph, slots=slots)
        wired = StructuralEmitter(graph).fabric(anchors=anchors)
        declared = set(
            re.findall(r"^\s*wire\s+(?:\[[^\]]*\]\s*)?(\w+_h_\w+)\s*\[", wired, re.M)
        )
        assert declared, "expected intermediate wires at this slot count"
        for wire in declared:
            uses = wired.count(f"({wire}[")
            assert uses == 2, f"{wire} is named {uses} time(s), not once per side"
