# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The unit HLS synthesises once, and how it meets the fabric.

The array program the dataflow path emits cannot be synthesised at all: every
site stores into the same result tensor, and HLS dataflow permits one writer per
array, so `csynth` rejects it at any grid size. A role built on its own -- every
port a stream, results included -- synthesises in seconds and is reused across
every site of its class.

What has to hold for that reuse to be sound is checked here: the unit must
compute what its site computed, its sites must be interchangeable, and the
parameter names HLS invents must map back to the fabric's ports correctly.
"""

import shutil
import subprocess

import pytest

import allo.spmw as spmw
from allo.spmw import rtl
from allo.spmw.role_ip import (
    UnitEmitter,
    build_unit,
    check_wrapper,
    unit_interface,
    wrapper_sv,
)

from test_spmw_attention import attention_pv
from test_spmw_fft import fft_spatial
from test_spmw_rolled import gemm_of
from test_spmw_tiled import tiled_gemm
from test_spmw_tpu import tpu_matmul

# Every role of these builds as a unit. The TPU and attention only do so because
# a memory port the unit *reads* becomes a value it holds: in the array a site
# reads the parent's weight tensor at its own coordinates, `local_W[i, j]`,
# which is per-site data rather than coordinate-dependent computation.
BUILDS = {"tpu": tpu_matmul, "attention": attention_pv(2), "tiled": tiled_gemm}


@pytest.mark.parametrize("size", [3, 4, 6])
def test_every_role_becomes_a_unit(size):
    graph = spmw.elaborate(gemm_of(size))
    emitter = UnitEmitter(graph)
    placement = emitter.placements()[0]
    for order in range(len(emitter.classes(placement))):
        text, _extras = emitter.program(placement, order)
        assert "@df.kernel" in text
        assert emitter.role_name(placement, order) in text


def test_a_unit_takes_only_streams():
    """Including its result: a memory port would make it a controlled IP.

    This is the difference between synthesisable and not, not a nicety -- the
    shared result tensor is exactly what HLS dataflow rejects.
    """
    graph = spmw.elaborate(gemm_of(4))
    emitter = UnitEmitter(graph)
    placement = emitter.placements()[0]
    text, _ = emitter.program(placement, 2)
    for line in text.splitlines():
        if line.strip().startswith(("c:", "west:", "north:", "east:", "south:")):
            assert "Stream[" in line, line
    assert "c[0].put(" in text, "the result must leave on a stream"


def test_the_unit_keeps_the_arithmetic():
    """The body is carried through, so the unit computes what the site did."""
    graph = spmw.elaborate(gemm_of(4))
    emitter = UnitEmitter(graph)
    text, _ = emitter.program(emitter.placements()[0], 2)
    assert "acc += a * b" in text
    # `size` stays a name and is resolved from the body's captured environment
    # when the program is imported -- the same way the array program does it, so
    # a unit written against module-level constants still compiles.
    assert "range(size)" in text
    assert emitter.low.injected.get("size") == 4


def test_the_ports_match_the_fabric_stub():
    """The IP and the black box the array instantiates must be one interface."""
    graph = spmw.elaborate(gemm_of(4))
    unit = UnitEmitter(graph)
    struct = rtl.StructuralEmitter(graph)
    placement = unit.placements()[0]
    for order in range(len(unit.classes(placement))):
        assert [p.name for p, _f in unit.ports(placement, order)] == [
            p.name for p, _f in struct.unit_ports(placement, order)
        ]


def test_a_coordinate_dependent_role_is_refused():
    """A role whose sites are not interchangeable cannot be one IP.

    Silence here would freeze the unit to whichever site was listed first and
    run it everywhere else -- wrong numbers, no error.
    """
    graph = spmw.elaborate(gemm_of(4))
    emitter = UnitEmitter(graph)
    placement = emitter.placements()[0]
    order = 2
    _sig, _routing, sites = emitter.classes(placement)[order]
    source, pids, _rw = emitter._body(placement, order, sites[0])
    import ast

    text = ast.unparse(ast.Module(body=source, type_ignores=[]))
    with pytest.raises(Exception, match="coordinates"):
        emitter._check_uniform(placement, order, text + f"\n{pids[0]}", pids)


@pytest.mark.parametrize("name", sorted(BUILDS))
def test_every_role_of_every_design_builds(name):
    graph = spmw.elaborate(BUILDS[name])
    emitter = UnitEmitter(graph)
    built = 0
    for placement in emitter.placements():
        for order in range(len(emitter.classes(placement))):
            text, _extras = emitter.program(placement, order)
            assert "@df.kernel" in text
            built += 1
    assert built > 0


def test_a_stationary_weight_is_held_not_indexed():
    """`local_W[i, j]` is per-site data, so the unit takes it on a port.

    Reading it once and holding it is what `stationary` already means, and it is
    what lets a role whose sites differ only in *data* be one IP.
    """
    graph = spmw.elaborate(tpu_matmul)
    emitter = UnitEmitter(graph)
    placement = emitter.placements()[0]
    text, _ = emitter.program(placement, 0)
    assert "_st_w" in text, text
    assert ".get()" in text.split("_st_w", 1)[1].splitlines()[0]
    assert "local_W" not in text, "the parent's tensor must not reach the unit"


def test_a_genuinely_positional_role_is_refused():
    """The FFT butterfly reads its stage and index, so it is not one unit.

    A single-instance kernel's pid is always zero, so compiling this would run
    every site as if it were the origin -- wrong numbers, no error. It is
    refused until a unit can take its coordinates as inputs.
    """
    graph = spmw.elaborate(fft_spatial)
    emitter = UnitEmitter(graph)
    placement = emitter.placements()[0]
    with pytest.raises(Exception, match="grid coordinates"):
        emitter.program(placement, 0)


# -- the toolchain ----------------------------------------------------------


def _vitis():
    return shutil.which("vitis_hls")


@pytest.mark.skipif(_vitis() is None, reason="Vitis HLS not on PATH")
def test_the_parameter_mapping_is_recovered_and_checked():
    """HLS renames every parameter; the mapping back is derived, then verified.

    Position alone would be a guess, so it is cross-checked against whether the
    body reads or writes each parameter. A slipped mapping would wire a reader
    onto a writer's FIFO.
    """
    graph = spmw.elaborate(gemm_of(4))
    emitter = UnitEmitter(graph)
    placement = emitter.placements()[0]
    built = build_unit(graph, placement, 2, target="vhls")
    ports = emitter.ports(placement, 2)
    mapping = unit_interface(str(built.hls_code), "pe_r2", ports)
    assert [p.name for _param, p in mapping] == ["west", "north", "east", "c"]
    for _param, port in mapping:
        assert port.direction in ("in", "out")


@pytest.mark.skipif(_vitis() is None, reason="Vitis HLS not on PATH")
def test_a_wrong_mapping_is_caught_not_carried():
    graph = spmw.elaborate(gemm_of(4))
    emitter = UnitEmitter(graph)
    placement = emitter.placements()[0]
    built = build_unit(graph, placement, 2, target="vhls")
    ports = list(reversed(emitter.ports(placement, 2)))
    with pytest.raises(Exception, match="mapping is wrong|cannot be confirmed"):
        unit_interface(str(built.hls_code), "pe_r2", ports)


@pytest.mark.skipif(_vitis() is None, reason="Vitis HLS not on PATH")
def test_the_wrapper_elaborates(tmp_path):
    """The shim that gives the IP the fabric's port names must itself compile."""
    if shutil.which("xvlog") is None:
        pytest.skip("Vivado xvlog not on PATH")
    graph = spmw.elaborate(gemm_of(4))
    emitter = UnitEmitter(graph)
    placement = emitter.placements()[0]
    built = build_unit(graph, placement, 2, target="vhls")
    code = str(built.hls_code)
    text = wrapper_sv(graph, placement, 2, code)
    # A stub with the IP's real interface, built from the recovered mapping, so
    # the wrapper elaborates without waiting on a synthesis run.
    signals = ["ap_clk", "ap_rst"]
    for param, port in unit_interface(code, "pe_r2", emitter.ports(placement, 2)):
        tail = (
            ("dout", "empty_n", "read")
            if port.direction == "in"
            else (
                "din",
                "full_n",
                "write",
            )
        )
        signals += [f"{param}_{s}" for s in tail]
    src = tmp_path / "wrap.sv"
    src.write_text(
        text
        + "\nmodule pe_r2_0 (\n"
        + ",\n".join(f"  inout wire [31:0] {s}" for s in signals)
        + "\n);\nendmodule\n"
    )
    done = subprocess.run(
        ["xvlog", "-sv", str(src)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_check_wrapper_rejects_a_port_the_ip_lacks():
    """The wrapper is written before synthesis; this closes that gap."""
    wrapper = "module m (\n);\n  m_0 u (.ap_clk(ap_clk), .v9_din(x));\nendmodule\n"
    exported = "module m_0 (\n  ap_clk,\n  v0_din\n);\nendmodule\n"
    with pytest.raises(Exception, match="does not have"):
        check_wrapper(wrapper, exported)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
