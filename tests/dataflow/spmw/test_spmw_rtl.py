# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import shutil
import subprocess
import tempfile

import pytest
import allo.spmw as spmw
import allo.backend.hls as hls
from allo.spmw_rtl import (
    emit_structural_verilog,
    emit_role_ip,
    emit_role_ip_project,
    emit_cosim_project,
    emit_vitis_rtl_project,
)
from allo.ir.types import float32


def _systolic_twin(M, N, K):
    grid = spmw.mesh((M, N))

    @spmw.unit
    def pe(ctx):
        c: float32 = 0
        for k in range(K):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    return gemm


def _dangling_role_region(M=4, N=4, K=4):
    grid = spmw.mesh((M, N))

    @spmw.unit
    def pe(ctx):
        ctx.west.get()  # dangling: reads `west`, never relays it, no stream feeds it

    @spmw.region()
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        spmw.map(pe, grid=grid)

    return gemm


def _module_types(sv):
    return set(re.findall(r"module (\w+)", sv))


def test_role_ip_export_enforces_strict_topology():
    """`emit_role_ip` / `emit_role_ip_project` are backend-facing exports: they must not emit `kernel.cpp`
    for a region with a dangling mesh boundary. Strict topology (task6.1) is enforced before emission.
    """
    with pytest.raises(spmw.SPMWError, match="unhandled"):
        emit_role_ip(_dangling_role_region())
    with pytest.raises(spmw.SPMWError, match="unhandled"):
        emit_role_ip_project(_dangling_role_region())


def test_structural_verilog_consumes_spmw_map():
    # the structural RTL top is derived from the rolled spmw.map (grid/families/roles off the IR): a
    # structural spmw_top that instantiates one role module per role inside generate loops over the
    # grid, wired by one FIFO per channel -- O(#roles) module text, not O(P0*P1)
    sv = emit_structural_verilog(_systolic_twin(4, 4, 4))
    assert "module spmw_top" in sv
    assert "generate" in sv and "genvar i, j" in sv
    # exactly one compute-role module and one instantiation site (rolled in a generate nest, not P^2
    # textual copies)
    assert sv.count("module pe_interior") == 1
    assert (
        sv.count("pe_interior #(.DW") == 1
    )  # the single instantiation (the def uses #(parameter)
    # one FIFO module, instantiated for both systolic families
    assert sv.count("module spmw_fifo") == 1
    assert "spmw_fifo #(.DW(DW), .DEPTH(DA))" in sv
    assert "spmw_fifo #(.DW(DW), .DEPTH(DB))" in sv
    # no per-grid-point module names leaked in
    assert not re.search(r"module \w+_\d+_\d+", sv)
    # the systolic wiring: west reads fa[i][j], east writes fa[i][j+1]; north/south likewise on fb
    assert ".west_dout(fa_dout[i][j])" in sv
    assert ".east_din(fa_din[i][j + 1])" in sv
    assert ".north_dout(fb_dout[i][j])" in sv
    assert ".south_din(fb_din[i + 1][j])" in sv


def test_structural_module_types_constant_across_grid_sizes():
    # the same module TYPES at 4x4 and 8x8 -- O(#roles), constant as the grid scales; only the M/N
    # parameters differ, exactly the synthesis-time-win regularity the HLS body count shows
    small = emit_structural_verilog(_systolic_twin(4, 4, 4))
    large = emit_structural_verilog(_systolic_twin(8, 8, 4))
    assert _module_types(small) == _module_types(large)
    assert _module_types(small) == {
        "spmw_top",
        "spmw_fifo",
        "pe_interior",
        "load_a",
        "load_b",
        "drain",
        "collect",
    }
    assert "parameter M = 4" in small and "parameter M = 8" in large


def test_structural_verilog_per_family_fifo_depths():
    # a region with different declared depths per family threads the per-family FIFO depths into the
    # DA/DB parameters (A east/west -> DA, B north/south -> DB), floored at K
    grid = spmw.mesh((4, 4))

    @spmw.unit
    def pe(ctx):
        c: float32 = 0
        for k in range(4):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: float32[4, 4], B: float32[4, 4], C: float32[4, 4]):
        spmw.map(pe, grid=grid, depths={"east": 8, "west": 8, "north": 4, "south": 4})
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    sv = emit_structural_verilog(gemm)
    assert "parameter DA = 8" in sv  # A east/west family
    assert "parameter DB = 4" in sv  # B north/south family


@pytest.mark.skipif(shutil.which("xvlog") is None, reason="requires Vivado xvlog")
def test_structural_verilog_parses_with_xvlog():
    # the emitted structural top is syntactically valid SystemVerilog: Vivado's xvlog analyzes all six
    # module types clean (the RTL counterpart of the rolled HLS top being csynth-ready)
    sv = emit_structural_verilog(_systolic_twin(4, 4, 4))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "spmw_top.sv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(sv)
        result = subprocess.run(
            ["xvlog", "-sv", path],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    for name in (
        "spmw_fifo",
        "pe_interior",
        "load_a",
        "load_b",
        "drain",
        "collect",
        "spmw_top",
    ):
        assert f"analyzing module {name}" in result.stdout, result.stdout


def test_build_target_rtl_returns_structural_verilog():
    # the structural RTL path is reachable through the public build API
    sv = spmw.build(_systolic_twin(4, 4, 4), target="rtl")
    assert "module spmw_top" in sv
    assert sv == emit_structural_verilog(_systolic_twin(4, 4, 4))


def test_role_ip_is_free_running_all_stream():
    # the interior PE exports as a free-running ap_ctrl_none IP whose every port is a stream (the
    # result leaves on a c_out stream, no memref), so it is a pure ap_fifo dataflow block
    ip = emit_role_ip(_systolic_twin(4, 4, 4))
    assert "void pe_interior(" in ip
    assert "#pragma HLS interface ap_ctrl_none port=return" in ip
    assert "hls::stream<float> &c_out" in ip
    assert "c_out.write(" in ip
    assert "c_local" not in ip


@pytest.mark.skipif(
    not hls.is_available("vitis_hls"), reason="requires the Vitis HLS toolchain"
)
def test_role_ip_csynths_free_running_ap_fifo():
    # csynth the role IP: the generated top RTL is free-running (no ap_start/ap_done block-control
    # ports) and its stream ports are ap_fifo named exactly as the structural top's black box wires
    # them -- so the exported IP drops straight into spmw_top
    with tempfile.TemporaryDirectory() as tmp:
        emit_role_ip_project(_systolic_twin(4, 4, 4), project=tmp)
        subprocess.run(
            ["vitis_hls", "-f", "run.tcl"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
        rtl_path = os.path.join(
            tmp, "role_ip.prj", "solution1", "syn", "verilog", "pe_interior.v"
        )
        with open(rtl_path, encoding="utf-8") as handle:
            rtl = handle.read()
    header = re.search(r"module pe_interior \(([^;]*)\);", rtl, re.DOTALL).group(1)
    assert "ap_start" not in header and "ap_done" not in header
    for sig in ("west_dout", "east_din", "north_dout", "south_din", "c_out_din"):
        assert sig in header, header


def test_behavioral_top_is_self_contained():
    # behavioral mode fills the role modules with simulation bodies (an FP MAC PE, loaders, drains)
    # instead of black boxes, so the same spmw_top is a self-contained, simulatable design
    sv = emit_structural_verilog(_systolic_twin(2, 2, 2), mode="behavioral")
    assert "// role IP body: ap_ctrl_none export / synth" not in sv
    assert "$bitstoshortreal" in sv  # the FP MAC datapath in the PE
    assert "`timescale" in sv


@pytest.mark.skipif(shutil.which("xsim") is None, reason="requires Vivado xsim")
def test_structural_top_cosim_matches_oracle():
    # the behavioral structural top computes A@B: xsim of the self-contained spmw_top + a self-checking
    # testbench reports COSIM PASS (C matches the shortreal oracle) -- the RTL-path correctness gate.
    # A non-square 2x3x4 grid exercises the B-column feed (load_b gathers B[*][j], not row j).
    with tempfile.TemporaryDirectory() as tmp:
        emit_cosim_project(_systolic_twin(2, 3, 4), project=tmp)
        subprocess.run(
            ["xvlog", "-sv", "dut.sv", "tb.sv"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        subprocess.run(
            ["xelab", "tb", "-s", "sim"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        result = subprocess.run(
            ["xsim", "sim", "-R"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
    assert "COSIM PASS" in result.stdout, result.stdout


def test_vitis_rtl_project_reuses_one_role_ip():
    # the vitis_rtl packaging project pairs the structural top with a SINGLE reusable role IP: the
    # top instantiates one pe_interior across the grid, and the IP is synthesized/exported once
    files = spmw.build(_systolic_twin(4, 4, 4), target="vitis_rtl")
    assert set(files) == {
        "spmw_top.sv",
        "kernel.cpp",
        "synth_ip.tcl",
        "package.tcl",
        "build.sh",
    }
    # hierarchical IP reuse: the exported PE IP is instantiated exactly once -- inside the
    # pe_row/pe_col generate loop that stamps it across the grid -- via the HLS IP ABI
    # (ap_clk/ap_rst + ap_fifo stream ports), not the old parameterized-module form; its body is the
    # exported IP RTL, never redefined in the top.
    assert files["spmw_top.sv"].count("pe_interior") == 1
    assert "pe_interior u_pe (.ap_clk(clk), .ap_rst(~rst_n)" in files["spmw_top.sv"]
    assert "module pe_interior" not in files["spmw_top.sv"]
    # the reusable PE IP is synthesized+exported once, and its RTL is bound into the hierarchy
    assert "export_design -format ip_catalog" in files["synth_ip.tcl"]
    assert (
        "syn/verilog" in files["package.tcl"]
    )  # the PE IP RTL is added to the project
    assert "set_property top spmw_top" in files["package.tcl"]


def test_vitis_rtl_project_writes_files(tmp_path):
    files = emit_vitis_rtl_project(_systolic_twin(2, 2, 2), project=str(tmp_path))
    for name in files:
        assert (tmp_path / name).exists()


@pytest.mark.skipif(
    not hls.is_available("vitis_hls") or shutil.which("xvlog") is None,
    reason="requires the Vitis HLS toolchain + Vivado xvlog",
)
def test_vitis_rtl_assembles_reusable_ip_hierarchy():
    # the reusable PE IP synthesizes + exports as a real Vivado IP catalog package, and its RTL binds
    # into the synth structural top: xvlog analyzes spmw_top + the exported PE RTL together, so the
    # single PE IP fills the pe_interior slot across the grid -- a real synthesizable hierarchy
    import glob

    with tempfile.TemporaryDirectory() as tmp:
        emit_vitis_rtl_project(_systolic_twin(2, 2, 2), project=tmp)
        subprocess.run(
            ["vitis_hls", "-f", "synth_ip.tcl"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=900,
            check=True,
        )
        zips = glob.glob(
            os.path.join(tmp, "role_ip.prj", "solution1", "impl", "ip", "*.zip")
        )
        assert zips, "expected an exported IP catalog .zip package"
        pe_rtl = glob.glob(
            os.path.join(tmp, "role_ip.prj", "solution1", "syn", "verilog", "*.v")
        )
        result = subprocess.run(
            ["xvlog", "-sv", "spmw_top.sv", *pe_rtl],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
    assert "analyzing module pe_interior" in result.stdout, result.stdout
    assert "analyzing module spmw_top" in result.stdout, result.stdout
