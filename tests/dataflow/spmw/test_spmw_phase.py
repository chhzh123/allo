# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""task6.2 -- `spmw.phase()` writer epochs + single-writer verification (AC-7).

A `spmw.place` pins an operand to a `spmw.shared`/`banked` buffer. Two maps that both write that
buffer race unless they run in different `spmw.phase()` epochs; the strict backend validation rejects
concurrent (same-epoch or both-unphased) writers, and phase-separated writers pass with the epoch token
visible in the lowered IR.
"""

import pytest
import allo.spmw as spmw
from allo.ir.types import float32


def _two_writer_region(*, phased, same_phase=False, kind="shared"):
    """A region whose operand ``C`` is written by two mapped units, optionally split across phases.

    ``phased=False`` -> both writers unphased (concurrent); ``phased=True, same_phase=False`` -> each
    writer in its own phase (separated); ``same_phase=True`` -> both writers in one shared phase
    (still concurrent). ``kind`` selects a shared or banked target buffer.
    """
    grid = spmw.mesh((2, 2))

    @spmw.unit
    def w1(ctx):
        pass

    @spmw.unit
    def w2(ctx):
        pass

    def _buffer():
        if kind == "banked":
            return spmw.banked(float32[2, 2], on="row", banks=2)
        return spmw.shared(float32[2, 2], space="L2")

    @spmw.region()
    def r(C: float32[2, 2]):
        if same_phase:
            with spmw.phase("compute"):
                spmw.map(w1, grid=grid)
                spmw.map(w2, grid=grid)
        elif phased:
            with spmw.phase("first"):
                spmw.map(w1, grid=grid)
            with spmw.phase("second"):
                spmw.map(w2, grid=grid)
        else:
            spmw.map(w1, grid=grid)
            spmw.map(w2, grid=grid)
        spmw.stream_out(C, from_=w1, where="local", as_="cbuf")
        spmw.stream_out(C, from_=w2, where="local", as_="cbuf")
        spmw.place("C", _buffer())

    return r


def _single_writer_region():
    grid = spmw.mesh((2, 2))

    @spmw.unit
    def w(ctx):
        pass

    @spmw.region()
    def r(C: float32[2, 2]):
        spmw.map(w, grid=grid)
        spmw.stream_out(C, from_=w, where="local", as_="cbuf")
        spmw.place("C", spmw.shared(float32[2, 2], space="L2"))

    return r


def _phased_systolic_region():
    """A lowerable systolic GEMM whose single map runs in a named phase, for the IR-visibility check."""
    grid = spmw.mesh((3, 3))

    @spmw.unit
    def pe(ctx):
        c: float32 = 0
        for k in range(2):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: float32[3, 2], B: float32[2, 3], C: float32[3, 3]):
        with spmw.phase("compute"):
            spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    return gemm


def test_concurrent_unphased_writers_rejected():
    """Two maps writing the same shared operand with no phase are concurrent -> rejected at build."""
    with pytest.raises(spmw.SPMWError, match="concurrent writers"):
        spmw.check_topology(_two_writer_region(phased=False))
    with pytest.raises(spmw.SPMWError, match="concurrent writers"):
        spmw.lower(_two_writer_region(phased=False))


def test_same_phase_writers_rejected():
    """Two writers inside ONE phase are still concurrent (a phase groups, it does not separate)."""
    with pytest.raises(spmw.SPMWError, match="concurrent writers"):
        spmw.check_topology(_two_writer_region(phased=True, same_phase=True))


def test_banked_concurrent_writers_rejected():
    """Banking proves a single map's own accesses conflict-free; it cannot prove two maps write
    disjoint banks, so concurrent writers to a banked buffer are rejected too."""
    with pytest.raises(spmw.SPMWError, match="concurrent writers"):
        spmw.check_topology(_two_writer_region(phased=False, kind="banked"))


def test_phase_separated_writers_accepted():
    """The same two writers, each in its own phase, are temporally separated -> accepted."""
    spmw.check_topology(_two_writer_region(phased=True))


def test_single_writer_accepted():
    """One writer per shared buffer is always fine, phased or not."""
    spmw.check_topology(_single_writer_region())


def test_direct_store_concurrent_writers_rejected():
    """A unit can write an operand IN PLACE (`ctx.C[...] = ...`) with no stream_out -- the pipeline write
    form that `generate_pipeline_source` lowers to a kernel store. The single-writer check must count
    those direct stores too, or two unphased maps writing a placed shared buffer would race undetected.
    """
    grid = spmw.mesh((2, 2))

    @spmw.unit
    def w1(ctx):
        ctx.C[0, 0] = 1.0

    @spmw.unit
    def w2(ctx):
        ctx.C[1, 1] = 2.0

    @spmw.region()
    def r(C: float32[2, 2]):
        spmw.map(w1, grid=grid)
        spmw.map(w2, grid=grid)
        spmw.place("C", spmw.shared(float32[2, 2], space="L2"))

    with pytest.raises(spmw.SPMWError, match="concurrent writers"):
        spmw.check_topology(r)


def test_default_validate_is_permissive():
    """The permissive default `spmw.validate` does not run the single-writer gate (skeletal frontend
    regions stay usable); only the strict backend path enforces it."""
    spmw.validate(_two_writer_region(phased=False))


def test_phase_lowers_to_visible_ir_attr():
    """A map declared in a `spmw.phase()` carries the epoch token on the lowered `spmw.map` op, so the
    single-writer contract is visible in the IR the checking path/backends consume."""
    ir = str(spmw.lower(_phased_systolic_region()))
    assert "spmw.phase = 1 : i64" in ir


def test_phase_outside_region_rejected():
    """`spmw.phase()` only means something while a region body is being traced."""
    with pytest.raises(spmw.SPMWError, match="inside an @spmw.region"):
        with spmw.phase("orphan"):
            pass


# ---- nested region composition: phases must propagate into inlined child maps ----------------------
def _nested_writer_region(*, same_phase):
    """A parent region that inlines two child writer regions (distinct units) writing the same shared
    ``C``, either in one parent phase (concurrent) or two (separated)."""
    grid = spmw.mesh((2, 2))

    @spmw.unit
    def wa(ctx):
        pass

    @spmw.unit
    def wb(ctx):
        pass

    @spmw.region()
    def child_a(C: float32[2, 2]):
        spmw.map(wa, grid=grid)
        spmw.stream_out(C, from_=wa, where="local", as_="cbuf")

    @spmw.region()
    def child_b(C: float32[2, 2]):
        spmw.map(wb, grid=grid)
        spmw.stream_out(C, from_=wb, where="local", as_="cbuf")

    @spmw.region()
    def top(C: float32[2, 2]):
        if same_phase:
            with spmw.phase("compute"):
                child_a(C)
                child_b(C)
        else:
            with spmw.phase("first"):
                child_a(C)
            with spmw.phase("second"):
                child_b(C)
        spmw.place("C", spmw.shared(float32[2, 2], space="L2"))

    return top


def test_nested_phase_propagates_and_separates_writers():
    """A parent `spmw.phase()` stamps the inlined child map: two child writer regions invoked in two
    parent phases are temporally separated (accepted), and both epoch tokens survive into the IR.
    """
    spmw.check_topology(_nested_writer_region(same_phase=False))
    ir = str(spmw.lower(_nested_writer_region(same_phase=False)))
    assert "spmw.phase = 1 : i64" in ir
    assert "spmw.phase = 2 : i64" in ir


def test_nested_writers_in_one_parent_phase_rejected():
    """Two child writer regions inlined inside ONE parent phase are concurrent -> rejected: the parent
    phase propagates to both child maps, so they share an epoch."""
    with pytest.raises(spmw.SPMWError, match="concurrent writers"):
        spmw.check_topology(_nested_writer_region(same_phase=True))
