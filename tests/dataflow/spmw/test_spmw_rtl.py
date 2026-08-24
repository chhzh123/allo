# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The structural RTL path: HLS builds the unit, RTL builds the array.

The split these tests exist for is a cost claim: synthesis is paid per *role*,
and the array's size is paid only in elaboration.  So the numbers that matter
are that the role count stays flat while the instance count grows, and that the
wiring is still correct at every size.

Correctness here is checked against ``topology.channels`` -- the link list the
emitter never reads -- because a mis-wired array is the failure that produces
wrong numbers rather than an error.
"""

import shutil
import subprocess

import pytest

import allo.spmw as spmw
from allo.spmw import rtl

from test_spmw_attention import attention_pv
from test_spmw_fft import fft_spatial
from test_spmw_rolled import gemm_of
from test_spmw_tpu import tpu_matmul

SIZES = [2, 3, 4, 6]

# The designs the old branch's emitter refused, and the reason each one was out
# of its reach.
OTHERS = {
    "fft": (fft_spatial, "key links, so TABLE families and block tokens"),
    "tpu": (tpu_matmul, "two placements joined by link, and stationary weights"),
    "attention": (attention_pv(2), "an interior boundary, opened by a split axis"),
}


@pytest.mark.parametrize("size", SIZES)
def test_every_link_lands_on_one_channel(size):
    """Both ends of a link must compute the same family and the same index.

    Checked against the topology's own channel list, so the emitter's tables
    cannot agree with themselves and be wrong together.
    """
    graph = spmw.elaborate(gemm_of(size))
    assert rtl.check_netlist(graph) > 0


@pytest.mark.parametrize("size", SIZES)
def test_synthesis_cost_tracks_roles_not_sites(size):
    """The point of the split: roles go flat, instances grow."""
    cost = rtl.cost(spmw.elaborate(gemm_of(size)))
    assert cost["instances"] == size * size
    if size >= 3:
        assert cost["roles"] == 9


def test_roles_stay_flat_as_the_array_grows():
    """One synthesis per role at 3x3 is the same nine at 16x16."""
    counts = {n: rtl.cost(spmw.elaborate(gemm_of(n)))["roles"] for n in (3, 4, 6, 8)}
    assert set(counts.values()) == {9}, counts


@pytest.mark.parametrize("size", SIZES)
def test_every_site_is_instantiated_exactly_once(size):
    """No site may be dropped or duplicated between the roles."""
    text = rtl.emit_structural_verilog(spmw.elaborate(gemm_of(size)))
    for i in range(size):
        for j in range(size):
            assert text.count(f"_{i}_{j} (") == 1, (i, j)


def test_a_boundary_family_leaves_the_top():
    """Loaders and drains are the DMA edge, so they are ports, not FIFOs."""
    graph = spmw.elaborate(gemm_of(4))
    emitter = rtl.StructuralEmitter(graph)
    placement = emitter.placements()[0]
    assert emitter.boundary_families(placement), "the GEMM streams A and B in"
    text = emitter.fabric()
    head = text[: text.index(");")]
    for fam in emitter.boundary_families(placement):
        assert f"{fam.name}_din" in head or f"{fam.name}_dout" in head


@pytest.mark.parametrize("size", SIZES)
def test_the_loaders_reach_the_array(size):
    """A stream fed by a loader must actually connect to the edge sites.

    A site signature holds only the ports linked to a *peer*, so `west` at
    column 0 -- fed by `stream_in` -- is absent from it. Deriving the port list
    from the signature declared the input streams and connected nothing to them:
    legal Verilog, elaborates clean, and A and B never enter the array.
    """
    graph = spmw.elaborate(gemm_of(size))
    emitter = rtl.StructuralEmitter(graph)
    placement = emitter.placements()[0]
    text = emitter.fabric()
    for fam in emitter.boundary_families(placement):
        if not fam.name.endswith("_bind"):
            continue
        hits = text.count(f"{fam.name}_dout[") + text.count(f"{fam.name}_din[")
        assert hits >= size, f"{fam.name} reaches {hits} sites, expected {size}"


@pytest.mark.parametrize("size", SIZES)
def test_no_family_is_left_dangling(size):
    """The check that would have caught the loaders, run on every design."""
    assert rtl.check_no_dangling_family(spmw.elaborate(gemm_of(size))) > 0


@pytest.mark.parametrize("size", SIZES)
def test_the_result_leaves_the_array(size):
    """Every site's `c` must reach the top, or the fabric computes and discards.

    `c` is a *memory* port, so an emitter that wired only streams would drop it
    silently and still elaborate -- the array would run and produce nothing.
    """
    graph = spmw.elaborate(gemm_of(size))
    emitter = rtl.StructuralEmitter(graph)
    placement = emitter.placements()[0]
    result = next(f for f in emitter.memory_families(placement) if "_c_" in f.name)
    assert rtl._volume(result.shape) == size * size
    text = emitter.fabric()
    head = text[: text.index(");")]
    assert f"{result.name}_din" in head, "the result stream must be a top-level port"
    for i in range(size):
        for j in range(size):
            idx = emitter.channel_index(
                placement,
                (i, j),
                next(p for p in placement.iface.ports() if p.name == "c"),
                result,
            )
            assert f"{result.name}_din[{idx}]" in text


def test_a_shared_memory_port_is_refused_not_guessed():
    """`Mem` is random access; a stream would be a wrong answer, not a partial one."""
    graph = spmw.elaborate(gemm_of(3))
    emitter = rtl.StructuralEmitter(graph)
    placement = emitter.placements()[0]
    port = next(p for p in placement.iface.ports() if p.name == "c")
    original, port.access = port.access, "rw"
    try:
        emitter._mem_families.clear()
        with pytest.raises(Exception, match="shared random-access"):
            emitter.memory_family(placement, port)
    finally:
        port.access = original
        emitter._mem_families.clear()


def test_a_port_that_leaves_the_grid_is_refused():
    """Silence here would mis-wire channel zero; the old lookup did exactly that."""
    graph = spmw.elaborate(gemm_of(4))
    emitter = rtl.StructuralEmitter(graph)
    placement = emitter.placements()[0]
    fam = emitter.peer_families(placement)[0]
    port = next(p for p in placement.iface.ports() if p.name == "east")
    with pytest.raises(Exception, match="whose extent is"):
        emitter.channel_index(placement, (0, 99), port, fam)


# -- beyond the systolic mesh -----------------------------------------------
#
# `feat/spmw` had this same HLS-unit/RTL-fabric split and validated it on
# hardware, but its emitter took the wiring from a four-entry literal and raised
# NotImplementedError unless the design was a single-role systolic GEMM with
# exactly the east/west + north/south families. These are the three designs that
# gate would have refused; the netlist check is the same one, against each
# design's own link list.


@pytest.mark.parametrize("name", sorted(OTHERS))
def test_designs_the_old_gate_refused(name):
    fabric, _why = OTHERS[name]
    graph = spmw.elaborate(fabric)
    assert rtl.check_netlist(graph) > 0
    text = rtl.emit_structural_verilog(graph)
    assert "module spmw_top" in text


def test_a_key_linked_design_uses_slot_tables():
    """FFT's links are keyed, so its families are addressed by lookup.

    This is the case the old emitter could not express at all: its routing was a
    constant displacement per port name.
    """
    emitter = rtl.StructuralEmitter(spmw.elaborate(fft_spatial))
    placement = emitter.placements()[0]
    kinds = {f.kind for f in emitter.peer_families(placement)}
    assert "table" in kinds, kinds


def test_a_link_is_internal_not_a_boundary():
    """A `link` has both ends inside the fabric, so it needs FIFOs.

    `_plan_link` registers one family under *both* placements. Treating it as a
    boundary declared the same top-level port twice -- which the elaborator
    caught -- and would have left the MXU and the activation row unconnected.
    """
    emitter = rtl.StructuralEmitter(spmw.elaborate(tpu_matmul))
    internal, boundary = emitter.families()
    names = {f.name for f in internal}
    assert any(n.endswith("_bind") for n in names), names
    assert len(names) + len({f.name for f in boundary}) == len(internal) + len(boundary)


def test_two_placements_both_reach_the_fabric():
    """The TPU is an MXU plus an activation row, joined by `link`."""
    emitter = rtl.StructuralEmitter(spmw.elaborate(tpu_matmul))
    assert len(emitter.placements()) == 2
    text = emitter.fabric()
    for placement in emitter.placements():
        for name in emitter.role_names(placement):
            assert f"{name} u_{name}_" in text


# -- the real elaborator ----------------------------------------------------


def _xvlog():
    return shutil.which("xvlog")


def _elaborates(graph, tmp_path):
    src = tmp_path / "spmw_top.sv"
    src.write_text(rtl.emit_structural_verilog(graph))
    done = subprocess.run(
        ["xvlog", "-sv", str(src)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    elab = subprocess.run(
        ["xelab", "spmw_top", "-s", "top_sim"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert elab.returncode == 0, elab.stdout + elab.stderr


@pytest.mark.skipif(_xvlog() is None, reason="Vivado xvlog not on PATH")
@pytest.mark.parametrize("size", [3, 4])
def test_the_fabric_elaborates(size, tmp_path):
    """Vivado has to accept it; a hand-checked netlist is not a compiled one."""
    _elaborates(spmw.elaborate(gemm_of(size)), tmp_path)


@pytest.mark.skipif(_xvlog() is None, reason="Vivado xvlog not on PATH")
@pytest.mark.parametrize("name", sorted(OTHERS))
def test_the_other_designs_elaborate(name, tmp_path):
    fabric, _why = OTHERS[name]
    _elaborates(spmw.elaborate(fabric), tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# -- what the whole-array path needs to synthesise --------------------------


@pytest.mark.parametrize("size", [3, 4])
def test_the_array_program_partitions_its_tensors(size):
    """Without this the array compiles, simulates, and will not synthesise.

    Every site writes one element of `C`, and HLS dataflow permits a single
    writer per interface array unless its elements are independently
    addressable. `csynth` stops at "Argument 'v185' failed dataflow checking",
    which `csim` never shows because that is only a C++ compile -- so the gap is
    invisible until synthesis. Partitioning is what closes it, and `feat/spmw`
    emitted the same pragma for the same reason.
    """
    code = str(spmw.build(gemm_of(size), target="vhls").hls_code)
    assert "array_partition" in code
    top = code[code.index("void top(") :]
    for axis in (1, 2):
        assert f"complete dim={axis}" in top, f"dim={axis} not partitioned"


def test_the_partitioning_can_be_turned_off():
    """Kept switchable: it scalarises the interface arrays, which is a real cost."""
    code = str(spmw.build(gemm_of(3), target="vhls", partition=False).hls_code)
    assert "array_partition" not in code


# -- driving the fabric ------------------------------------------------------
#
# The array elaborating is not the array computing. The numeric check is a real
# xsim run against the exported IPs, which needs nine syntheses and so lives in
# `scripts/spmw_build_array.py --cosim`. What is tested here is the part that
# decides whether such a run means anything: the stimulus order.


@pytest.mark.parametrize("size", [3, 4])
def test_the_boundary_plan_is_the_design_s_own_order(size):
    """Channel i of the A loader must carry row i, and B's must carry column j.

    Derived from `binding.imap`, the same map the reference simulator evaluates,
    rather than from a hand-written guess at the order -- so a testbench built
    on it drives the array the way the design says.
    """
    plan = rtl.boundary_plan(spmw.elaborate(gemm_of(size)))
    west = next(v for k, v in plan.items() if "west" in k)
    north = next(v for k, v in plan.items() if "north" in k)
    result = next(v for k, v in plan.items() if "_c_" in k)

    assert west["tensor"] == "A" and west["direction"] == "in"
    for i, channel in enumerate(west["channels"]):
        assert channel == [(i, k) for k in range(size)], f"west[{i}]"
    assert north["tensor"] == "B" and north["direction"] == "in"
    for j, channel in enumerate(north["channels"]):
        assert channel == [(k, j) for k in range(size)], f"north[{j}]"
    assert result["tensor"] == "C" and result["direction"] == "out"
    assert sorted(c[0] for c in result["channels"]) == sorted(
        (i, j) for i in range(size) for j in range(size)
    )


def test_a_boundary_port_carries_only_its_own_half_of_the_handshake():
    """An inbound family has no `din`; an outbound one has no `dout`.

    Declaring all six elaborates but leaves half dangling, and a driver then
    cannot tell which side of the handshake it owns.
    """
    emitter = rtl.StructuralEmitter(spmw.elaborate(gemm_of(4)))
    head = emitter.fabric().split(");", 1)[0]
    assert "pe_west_bind_dout" in head and "pe_west_bind_din" not in head
    assert "pe_c_mem_din" in head and "pe_c_mem_dout" not in head


@pytest.mark.skipif(_xvlog() is None, reason="Vivado xvlog not on PATH")
@pytest.mark.parametrize("size", [3, 4])
def test_the_testbench_compiles(size, tmp_path):
    """The generated driver has to be legal SystemVerilog before it can judge."""
    import numpy as np

    from allo.spmw.cosim import render_testbench

    graph = spmw.elaborate(gemm_of(size))
    a = np.arange(size * size, dtype=np.float32).reshape(size, size) % 4
    want = {"C": (a @ a).astype(np.float32)}
    (tmp_path / "tb.sv").write_text(render_testbench(graph, {"A": a, "B": a}, want))
    (tmp_path / "top.sv").write_text(rtl.emit_structural_verilog(graph))
    done = subprocess.run(
        ["xvlog", "-sv", "top.sv", "tb.sv"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
