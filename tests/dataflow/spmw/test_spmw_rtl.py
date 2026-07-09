# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import shutil
import subprocess
import tempfile

import pytest
import allo.spmw as spmw
from allo.spmw_rtl import emit_structural_verilog
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


def _module_types(sv):
    return set(re.findall(r"module (\w+)", sv))


def test_structural_verilog_consumes_spmw_map():
    # the structural RTL top is derived from the rolled spmw.map (grid/families/roles off the IR): a
    # structural spmw_top that instantiates one role module per role inside generate loops over the
    # grid, wired by one FIFO per channel -- O(#roles) module text, not O(P0*P1)
    sv = emit_structural_verilog(_systolic_twin(4, 4, 4))
    assert "module spmw_top" in sv
    assert "generate" in sv and "genvar i, j;" in sv
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
    for name in ("spmw_fifo", "pe_interior", "load_a", "load_b", "drain", "spmw_top"):
        assert f"analyzing module {name}" in result.stdout, result.stdout


def test_build_target_rtl_returns_structural_verilog():
    # the structural RTL path is reachable through the public build API
    sv = spmw.build(_systolic_twin(4, 4, 4), target="rtl")
    assert "module spmw_top" in sv
    assert sv == emit_structural_verilog(_systolic_twin(4, 4, 4))
