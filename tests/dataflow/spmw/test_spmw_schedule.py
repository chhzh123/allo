# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduling: how a unit is compiled, as opposed to what it computes.

The measurement these exist to protect (systolic GEMM PE, K=16, xcu280 at
300 MHz): the loop sits at II=7 because its float accumulator is a loop-carried
recurrence, and II = adder latency + 1. Binding a shorter adder trades
combinational delay for interval, and `ii=4` is the fastest point that still
closes timing -- 1.75x, with fewer DSPs than the default and not one rounding
changed.

Nothing here alters what a design means. That is the point of keeping schedule
on its own axis.
"""

import ast

import pytest

import allo.spmw as spmw
from allo.spmw import schedule as sched

from test_spmw_rolled import gemm_of
from test_spmw_tpu import tpu_matmul

OUTPUT_STATIONARY = """
acc: float32 = 0
for k in range(16):
    a = west[0].get()
    b = north[0].get()
    acc += a * b
    east[0].put(a)
c[0].put(acc)
"""

# The mini-TPU's MAC: the partial sum arrives on a stream and leaves on one, so
# no value is carried between iterations and the loop is already free to run at
# II=1. This is why the TPU, attention and FFT need no help and the GEMM does.
OUTPUT_FLOWING = """
for m in range(6):
    a = a_in[0].get()
    p = p_in[0].get()
    p_out[0].put(p + a * _st_w)
"""


def test_a_held_accumulator_is_found():
    assert sched.accumulators(ast.parse(OUTPUT_STATIONARY)) == ["acc"]


def test_a_streamed_partial_sum_is_not_an_accumulator():
    """Nothing is carried, so there is no recurrence to shorten."""
    assert sched.accumulators(ast.parse(OUTPUT_FLOWING)) == []


def test_an_augmented_assignment_counts_as_a_read():
    """`acc += x` reads acc, but its target carries a Store context.

    Missing that found no accumulators at all, and the binding silently did
    nothing.
    """
    tree = ast.parse("t: float32 = 0\nfor i in range(4):\n    t += i\n")
    assert sched.accumulators(tree) == ["t"]


def test_a_name_first_written_inside_the_loop_is_not_carried():
    """It is dead on entry, so each iteration is independent."""
    tree = ast.parse("for i in range(4):\n    t = i * 2\n    out[0].put(t)\n")
    assert sched.accumulators(tree) == []


def test_binding_survives_the_provenance_comments():
    """Allo puts `// L17` after every statement.

    A regex anchored at end-of-line matched nothing, so the optimisation was
    generated, reported, and had no effect.
    """
    code = (
        "    float v13 = acc;\t// L16\n"
        "    float v14 = v13 + v12;\t// L17\n"
        "    acc = v14;\t// L18\n"
    )
    out, bound = sched.bind_recurrences(code, ["acc"], 3)
    assert bound == ["v14"]
    assert "bind_op variable=v14 op=fadd impl=fabric latency=3" in out
    assert out.index("bind_op") > out.index("float v14"), "must follow the declaration"


def test_binding_an_absent_accumulator_is_a_no_op():
    """A missed binding costs speed, not correctness, so it is not fatal."""
    out, bound = sched.bind_recurrences("float x = 1;\n", ["acc"], 3)
    assert bound == [] and out == "float x = 1;\n"


def test_the_interval_defaults_and_overrides():
    placement = spmw.elaborate(gemm_of(3)).placements[0]
    assert sched.interval(placement, default=0) == 0
    spmw.pipeline(placement, ii=4)
    assert sched.interval(placement) == 4
    spmw.pipeline(placement, ii=2)
    assert sched.interval(placement) == 2, "re-applying replaces rather than stacks"
    assert len([d for d in placement.schedule if d.kind == sched.PIPELINE]) == 1


def test_pipeline_rejects_what_it_cannot_schedule():
    placement = spmw.elaborate(gemm_of(3)).placements[0]
    for bad in (-1, 1.5, "1"):
        with pytest.raises(Exception, match="non-negative int"):
            spmw.pipeline(placement, ii=bad)
    with pytest.raises(Exception, match="applies to a placement"):
        spmw.pipeline(object(), ii=1)


def test_scheduling_does_not_change_the_design():
    """The spec is the frontend; a schedule is a compilation choice."""
    before = spmw.elaborate(gemm_of(3))
    after = spmw.elaborate(gemm_of(3))
    spmw.pipeline(after.placements[0], ii=4)
    from allo.spmw.lower_df import render_source

    assert render_source(before) == render_source(after)


@pytest.mark.parametrize("design", [gemm_of(3), tpu_matmul])
def test_every_design_reports_its_recurrences(design):
    """Whether a unit can reach II=1 is decided by its dataflow, not its size."""
    from allo.spmw.role_ip import UnitEmitter

    graph = spmw.elaborate(design)
    emitter = UnitEmitter(graph)
    placement = emitter.placements()[0]
    _sig, _routing, sites = emitter.classes(placement)[0]
    body, _pids, _rw = emitter.body_for(placement, 0, sites[0])
    tree = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    carried = sched.accumulators(tree)
    # the GEMM holds its result; the TPU streams it
    assert (carried == ["acc"]) == (graph.fabric.name != "tpu_matmul")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class _FakeName:
    def __init__(self, value):
        self.value = value


class _FakeOp:
    def __init__(self, value):
        self.name = _FakeName(value)


class _FakeSchedule:
    """Just enough of a Schedule to exercise the lookup."""

    def __init__(self, names):
        self.module = type(
            "M",
            (),
            {"body": type("B", (), {"operations": [_FakeOp(n) for n in names]})},
        )
        self.asked = []

    def get_loops(self, name):
        # What Allo does: a name the module lacks is a RuntimeError, not a
        # lookup miss. Asking for one at all is the bug this fake exists to
        # catch.
        self.asked.append(name)
        if name not in sched.function_names(self):
            raise RuntimeError(f"Function {name} not found")
        return type("Band", (), {"loops": {}})()


def test_a_missing_function_is_never_looked_up():
    """A unit is looked up under two names and only one of them exists.

    Catching whatever `get_loops` raises got this wrong: the list of exception
    types omitted `RuntimeError`, which is the one Allo actually raises, and the
    whole build died the first time a real interval was asked for. Membership is
    checked against the module's own symbols instead.
    """
    schedule = _FakeSchedule(["pe_r0_0", "top"])
    sched.apply(schedule, ["pe_r0", "pe_r0_0"], 4)
    assert "pe_r0" not in schedule.asked, "asked for a function the module lacks"
    assert schedule.asked == ["pe_r0_0"], schedule.asked


def test_no_interval_asks_nothing():
    schedule = _FakeSchedule(["pe_r0_0"])
    assert sched.apply(schedule, ["pe_r0_0"], 0) == []
    assert schedule.asked == []


def test_function_names_reads_the_module():
    assert sched.function_names(_FakeSchedule(["a", "b"])) == ["a", "b"]


def test_a_unit_that_carries_nothing_is_left_at_peak():
    """A recurrence budget only means something if there is a recurrence.

    Forcing ii=4 on attention -- whose partial sum arrives on a stream -- took
    its MAC from II=1 to II=4. Asking for a wider interval can only cost a
    design that has nothing to trade.
    """
    from allo.spmw.role_ip import UnitEmitter

    graph = spmw.elaborate(tpu_matmul)
    emitter = UnitEmitter(graph)
    placement = emitter.placements()[0]
    _sig, _routing, sites = emitter.classes(placement)[0]
    body, _pids, _rw = emitter.body_for(placement, 0, sites[0])
    tree = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    assert sched.accumulators(tree) == [], "the TPU streams its partial sum"


# -- why II=1 is out of reach for an output-stationary float PE --------------
#
# Measured on the same 16-iteration loop, three ways (xcu280, 300 MHz):
#
#     arithmetic   carries?          II   iteration latency
#     float        yes, distance 1    7   14
#     int          yes, distance 1    1    5
#     float        no                 1   14
#
# Removing either condition gives II=1, so the constraint is a float add inside
# a distance-1 cycle -- not the systolic structure, and not float on its own.
# The bound is II >= latency of the recurrence / dependence distance.


def test_the_two_conditions_for_a_binding_recurrence():
    """A carried value only costs II when something slow is in the cycle.

    The detector answers the first half -- is a value carried at all -- which is
    what decides whether a recurrence budget can buy anything. The second half
    (how slow the operation is) is the adder's latency, and is what
    `bind_recurrences` trades against.
    """
    carried = ast.parse(OUTPUT_STATIONARY)
    flowing = ast.parse(OUTPUT_FLOWING)
    assert sched.accumulators(carried) == ["acc"]
    assert sched.accumulators(flowing) == []

    # An integer accumulator is still carried: the detector is about dataflow,
    # not types. It reaches II=1 because integer add is single-cycle, which is a
    # property of the adder rather than of the loop.
    integer = ast.parse(
        "acc: int32 = 0\nfor k in range(16):\n"
        "    acc += a_in[0].get() * b_in[0].get()\nc[0].put(acc)\n"
    )
    assert sched.accumulators(integer) == ["acc"]
