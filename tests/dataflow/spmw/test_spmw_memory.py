# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The array's own memory interface -- what makes it an accelerator, not a core.

Everywhere else here a fabric ends at its edge streams and something outside
holds the operands.  That is enough to measure the array, and it is what the
cosim testbench drives, but it is not what AutoSA builds: their kernel reaches
DRAM itself, through ``A_IO_L3_in`` and the AXI masters behind it.  Comparing
compute cycles against a design that also pays for its loads is not a
comparison, so the fabric needs the same thing.

A *mover* is the loader or drain a binding already implies -- ``stream_in``
synthesises one, so does ``gather``.  Until now it existed only inside the array
program.  Built on its own it becomes an IP whose tensor argument is an AXI
master, and :meth:`allo.spmw.rtl.StructuralEmitter.fabric` with ``memory=True``
instantiates those in place of the edge ports.

The number that matters is not how many movers there are -- that is fixed per
binding, so the compile cost stays flat -- but how many *instances* each has,
because an instance is a DRAM port:

| design | what feeds the array | masters at 16x16 |
|---|---|---|
| `gemm8` | one loader per row and column | 32 |
| `daisy` | the same, plus a per-column drain | 48 |
| `autosa` | two chains and a per-column drain | 18 |

A U280 has 32 HBM channels.  The first two do not fit at 16x16; the third does,
and it is the *chain* -- an ordinary SPMW placement, six lines of design -- that
makes the difference.  That is the point these tests pin down.
"""

import pytest

import allo.spmw as spmw
from allo.spmw import rtl
from allo.spmw.abi import axi_signals
from allo.spmw.errors import SPMWBindingError
from test_spmw_autosa_match import autosa_match_of
from test_spmw_daisy import daisy_of
from test_spmw_gemm_int8 import gemm_int8_of


def _emitter(design, size):
    return rtl.StructuralEmitter(spmw.elaborate(design(size)))


def _masters(emitter):
    return emitter.movers.masters()


def test_a_chain_costs_one_master_per_operand_whatever_the_size():
    """The head of a distribution chain is one instance at every grid size.

    This is the whole reason a chain is worth writing: the array behind it grows
    and its DRAM port does not.
    """
    for size in (4, 8, 16):
        emitter = _emitter(autosa_match_of, size)
        loaders = [
            len(emitter.movers.instances(i))
            for i in range(len(emitter.movers))
            if "load" in emitter.movers.name(i)
        ]
        assert loaders == [1, 1], (size, loaders)


def test_without_a_chain_the_masters_grow_with_the_array():
    """A plain mesh needs a DRAM port per row and per column.

    The control for the test above. Nothing about `gemm8` is wrong -- it is the
    same arithmetic in the same nine roles -- but its operands enter on N edge
    streams, so giving it a memory interface would want 2N masters.
    """
    for size in (4, 8, 16):
        emitter = _emitter(gemm_int8_of, size)
        assert _masters(emitter) == 2 * size, size
    assert _masters(_emitter(autosa_match_of, 16)) == 18  # 1 + 1 + 16


def test_the_drain_is_still_one_master_per_column():
    """What SPMW has not chained yet, stated rather than glossed.

    AutoSA has three levels on the way out -- per-PE, per-column, then one AXI
    master -- and this has the first two. The last column of the table in the
    module docstring is 18 rather than 3 because of exactly this.
    """
    emitter = _emitter(autosa_match_of, 16)
    drains = [
        len(emitter.movers.instances(i))
        for i in range(len(emitter.movers))
        if "drain" in emitter.movers.name(i)
    ]
    assert drains == [16]


def test_the_compile_cost_does_not_grow():
    """One mover *program* per binding -- the number of extra HLS runs.

    Instances are replicated in RTL, as role instances are, so adding the memory
    interface costs a fixed three synthesis runs at any size. This is what keeps
    the flat compile time flat.
    """
    counts = {size: len(_emitter(autosa_match_of, size).movers) for size in (4, 8, 16)}
    assert set(counts.values()) == {3}, counts


def test_the_head_of_a_chain_takes_no_coordinate():
    """A single-site bundle has no position to be told.

    It is not a cosmetic difference: the fabric wired a `_pid0` onto an IP that
    had no such port, and Vivado rejected the whole design.
    """
    emitter = _emitter(autosa_match_of, 4)
    head = next(
        i for i in range(len(emitter.movers)) if "feed_up" in emitter.movers.name(i)
    )
    drain = next(
        i for i in range(len(emitter.movers)) if "drain" in emitter.movers.name(i)
    )
    assert emitter.movers.shape(head) == ()
    assert [coords for _pos, coords, _s, _c in emitter.movers.instances(head)] == [[]]
    assert emitter.movers.shape(drain) == (4,)


def test_each_instance_owns_one_channel_of_its_family():
    """The mover and the sites must land on the same FIFO.

    Both ends ask `channel_index`, so this checks the answers are a permutation
    -- one channel each, none shared, none missed.
    """
    emitter = _emitter(autosa_match_of, 4)
    for index in range(len(emitter.movers)):
        channels = [c for _p, _co, _s, c in emitter.movers.instances(index)]
        assert sorted(channels) == list(range(len(channels))), (index, channels)


def test_the_memory_fabric_declares_a_master_per_instance():
    """The generated top, read back."""
    emitter = _emitter(autosa_match_of, 4)
    fabric = emitter.fabric(memory=True)
    assert "input  wire ap_start" in fabric
    assert "output wire ap_done" in fabric
    offsets = [l for l in fabric.splitlines() if l.strip().endswith("_offset,")]
    assert len(offsets) == _masters(emitter) == 6
    # every AXI signal of every master, and nothing dangling
    for suffix in ("ARVALID", "RDATA", "AWADDR", "BRESP"):
        assert fabric.count(f"_{suffix}") >= 6 * 2, suffix


def test_a_gathered_memory_port_has_no_mover_and_says_so():
    """`gemm8` writes its result through a `MemOut`, not a drain.

    There is no transfer to build an AXI master out of, so a memory-mapped
    fabric is refused rather than emitted with a channel that still has to be
    driven from outside -- which would look like an accelerator and not be one.
    """
    emitter = _emitter(gemm_int8_of, 4)
    with pytest.raises(SPMWBindingError, match="no mover"):
        emitter.fabric(memory=True)
    # the daisy chain drains on a stream, so it does have one
    _emitter(daisy_of, 4).fabric(memory=True)


def test_the_axi_table_is_a_complete_master():
    """Every AXI4 channel, and the widths that depend on the data width."""
    signals = dict((n, (d, w)) for n, d, w in axi_signals("gmem", 512))
    assert signals["m_axi_gmem_RDATA"] == ("input", 512)
    assert signals["m_axi_gmem_WSTRB"] == ("output", 64)
    assert signals["m_axi_gmem_ARADDR"] == ("output", 64)
    # read and write address channels, data both ways, and both responses
    for channel in ("AW", "AR"):
        for signal in ("VALID", "READY", "ADDR", "LEN", "SIZE", "BURST"):
            assert f"m_axi_gmem_{channel}{signal}" in signals
    assert signals["m_axi_gmem_ARREADY"][0] == "input"
    assert signals["m_axi_gmem_ARVALID"][0] == "output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _bench(size=4, latency=0):
    """A memory bench for the matched design, without running a simulator."""
    import numpy as np

    from allo.spmw.dram import MemoryBench

    graph = spmw.elaborate(autosa_match_of(size))
    data = {
        t.name: np.zeros(t.shape, dtype=np.int32 if str(t.dtype) == "i32" else np.int8)
        for t in graph.tensors.values()
    }
    return MemoryBench(graph, data, data, latency=latency)


def test_every_master_gets_a_memory_and_a_direction():
    """Two read masters and four written ones, at 4x4."""
    masters = _bench().masters()
    assert len(masters) == 6
    assert sum(1 for m in masters if m["reads"]) == 2
    assert {m["tensor"] for m in masters if m["reads"]} == {"At", "Bt"}
    assert {m["tensor"] for m in masters if not m["reads"]} == {"Ct"}


def test_a_drain_is_only_checked_where_it_writes():
    """Each instance fills its own slice of the result.

    Its memory is private, so everything else in it stays uninitialised.
    Comparing the whole tensor against one master reported twelve of Ct's
    sixteen elements as wrong when the design was in fact correct.
    """
    masters = _bench().masters()
    drains = [m for m in masters if not m["reads"]]
    for master in drains:
        assert len(master["touches"]) == 4, master["name"]
    # together they cover the result exactly once
    seen = [tuple(i) for m in drains for i in m["touches"]]
    assert len(seen) == len(set(seen)) == 16


def test_the_bench_starts_the_array_and_waits_for_one_done():
    """One start and one completion, whatever the port count."""
    text = _bench().render()
    assert text.count(".ap_start(start)") == 1
    assert text.count(".ap_done(done)") == 1
    assert "spmw_axi_ram" in text
    assert text.count("spmw_axi_ram #(") == 6


def test_the_latency_is_recorded_with_the_cycle_count():
    """A cycle count that does not say what memory it assumed is not a result."""
    assert "latency=64" in _bench(latency=64).render()
