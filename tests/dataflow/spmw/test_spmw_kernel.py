# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The wrapper that makes the array something a card can load.

Every check here is for a bug that actually happened, on hardware, where the
only diagnosis available was "the kernel did not finish" -- pyxrt exposes no way
to read a kernel's control map, so a stall on the card says nothing at all. Each
one cost a build to find and is cheap to keep.
"""

import re

import allo.spmw as spmw
from allo.spmw.kernel import ARG_BASE, arguments, control_sv, kernel_sv, kernel_xml
from allo.spmw.shell import _dma_name, families

from test_spmw_transformer import BIG


def _kernel():
    graph = spmw.elaborate(BIG.engine)
    args = arguments(graph)
    widths = {_dma_name(f): 32 for f in families(graph)}
    return graph, args, widths, kernel_sv(graph, args, widths)


def test_each_feeder_is_started_exactly_once():
    """An `ap_ctrl_hs` IP restarts if `ap_start` is still high when it finishes.

    The kernel's start has to stay high until the *slowest* feeder is done, so
    every faster one was taking the job a second time: two weights where the
    family has one, 21 program words where it has 17. The array survived being
    over-fed for a while and then stalled, with the extra tokens the only clue.

    So the start must be a latch cleared by that feeder's own `ap_ready`, not a
    combinational function of `ap_start`. Gating on `ap_done` is not enough --
    the IP accepts on `ap_ready` and asserts both in the same cycle.
    """
    _graph, _args, _widths, sv = _kernel()
    starts = re.findall(r"wire start_(\d+) = ([^;]+);", sv)
    assert starts, "every feeder should have its own start"
    for index, expr in starts:
        assert expr.strip() == f"pend[{index}]", expr
    # The latch itself: raised by a one-shot, dropped when that feeder accepts.
    assert "wire kick = ap_start & ~ap_start_d;" in sv
    for index, _expr in starts:
        assert f"if (kick) pend[{index}] <= 1'b1;" in sv
        assert f"else if (ready_{index}) pend[{index}] <= 1'b0;" in sv


def test_each_edge_fifo_holds_a_whole_pass():
    """A feeder is sequential; a systolic array is skewed.

    The feeder writes every channel's step-t token before any channel's t+1,
    but the last row of the mesh is fifteen steps behind the first. With a
    shallow FIFO the last channel fills, the feeder blocks on it, and the first
    row starves waiting for a token that cannot arrive until the last row moves
    -- which needs the first row to move. Deadlock, every FIFO empty.

    The depth has to be the family's own step count, so a whole pass fits.
    """
    graph, _args, _widths, sv = _kernel()
    plans = {f["name"]: f for f in families(graph)}
    depths = dict(re.findall(r"// family (\w+):[^\n]*?(\d+) deep", sv)) or dict(
        re.findall(r"// (\w+): \d+ channel\(s\)[^\n]*?, (\d+) deep", sv)
    )
    assert depths, "the emitted comment should record each edge FIFO's depth"
    for name, plan in plans.items():
        assert name in depths, name
        assert int(depths[name]) == max(2, plan["steps"]), name
    # And at least one family is deeper than a token or two, or the test above
    # would pass on a design where the bug could not bite.
    assert max(int(d) for d in depths.values()) >= 16


def test_a_completion_survives_being_read_in_the_same_cycle():
    """`ap_done` is one cycle wide, and the host polls continuously.

    Clearing `r_done` on a read of the control word without letting a
    simultaneous `ap_done` win loses the completion for good: the host then
    polls a finished kernel for ever. Per invocation the odds are small; a
    BERT-base layer is 227,328 invocations.
    """
    _graph, args, _widths, _sv = _kernel()
    ctrl = control_sv(args)
    set_at = ctrl.index("if (ap_done)        r_done <= 1'b1;")
    clear_at = ctrl.index("else if (ctrl_read) r_done <= 1'b0;")
    assert set_at < clear_at, "the clear must be the `else` of the set"
    # The read reports a same-cycle completion as well as the latched one.
    assert "r_done | ap_done" in ctrl
    # And a write is not accepted while a response is still outstanding.
    assert "wire          w_fire = AWVALID && WVALID && !b_valid;" in ctrl


def test_the_control_map_and_the_kernel_xml_agree():
    """Two descriptions of one register map; a drift here is silent.

    XRT writes arguments where `kernel.xml` says, and the slave decodes where
    `control_sv` says. If they disagree the kernel runs with whatever was left
    in the wrong register.
    """
    _graph, args, widths, _sv = _kernel()
    xml = kernel_xml(args, widths)
    ctrl = control_sv(args)
    for arg in args:
        assert f'name="{arg.name}"' in xml
        assert f'offset="{arg.offset:#06x}"' in xml, arg.name
        # The slave decodes the same address.
        assert re.search(rf"h0*{arg.offset:x}: r_{arg.name}", ctrl), arg.name
    offsets = [a.offset for a in args]
    assert len(set(offsets)) == len(offsets), "two arguments share an address"
    assert min(offsets) >= ARG_BASE, "an argument sits in the control words"
    assert all(o % 4 == 0 for o in offsets)


def test_a_pointer_gets_both_of_its_halves():
    """A 64-bit address is two registers, and forgetting the high one works
    on any machine whose buffers happen to sit below 4 GB."""
    _graph, args, _widths, _sv = _kernel()
    ctrl = control_sv(args)
    for arg in args:
        if not arg.pointer:
            continue
        assert re.search(rf"h0*{arg.offset:x}: r_{arg.name}\[31:0\]", ctrl)
        assert re.search(rf"h0*{arg.offset + 4:x}: r_{arg.name}\[63:32\]", ctrl)


def test_the_feeders_take_the_side_of_the_edge_the_fabric_leaves():
    """A feeder writes what the array reads, and reads what the array wrote.

    Getting this backwards elaborates -- both sides are just wires -- and then
    nothing ever moves. The feeder names its ports per channel (`out_3_din`),
    the fabric takes whole arrays (`mac_a_in_bind_dout`), so the two ends of
    one edge look nothing alike and a swap would not stand out by eye.
    """
    graph, _args, _widths, sv = _kernel()
    for plan in families(graph):
        name = plan["name"]
        if plan["reads"]:
            # feeder writes channel 0 ...
            assert f"_din({name}_e_din[0])" in sv, name
            assert f"_write({name}_e_write[0])" in sv, name
            # ... and the array reads the whole family.
            assert f".{name}_dout({name}_e_dout)" in sv, name
            assert f".{name}_read({name}_e_read)" in sv, name
        else:
            assert f"_dout({name}_e_dout[0])" in sv, name
            assert f"_read({name}_e_read[0])" in sv, name
            assert f".{name}_din({name}_e_din)" in sv, name
            assert f".{name}_write({name}_e_write)" in sv, name


def test_the_drain_is_the_only_family_the_array_writes():
    """One outbound family, and the checks above split on that flag.

    If every family read the same way, the test above would be asserting one
    branch twice and the other never.
    """
    graph, _args, _widths, _sv = _kernel()
    plans = families(graph)
    outbound = [f for f in plans if not f["reads"]]
    inbound = [f for f in plans if f["reads"]]
    assert len(outbound) == 1, "the engine drains through one family"
    assert len(inbound) >= 4, "and is fed by several"
