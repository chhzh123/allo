# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import tempfile

import allo.spmw as spmw
from allo.spmw_hls import emit_role_hls, emit_rolled_hls
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


def test_emit_role_hls_transcribes_the_datapath():
    cpp = emit_role_hls(_systolic_twin(4, 4, 4))
    assert "#include <hls_stream.h>" in cpp
    assert "void pe_interior(" in cpp
    assert "hls::stream<float>" in cpp
    # the port I/O becomes stream read/write and the MAC is transcribed verbatim
    assert ".read()" in cpp
    assert ".write(" in cpp
    assert "c += (a * b)" in cpp
    # one role body -- not one per grid point
    assert cpp.count("void pe_interior(") == 1


def _role_bodies(cpp):
    """The set of function names defined in an emitted HLS file."""
    return set(re.findall(r"\bvoid\s+(\w+)\s*\(", cpp))


def test_emit_rolled_hls_is_a_dataflow_top():
    cpp = emit_rolled_hls(_systolic_twin(4, 4, 4))
    # the rolled top calls each role in unrolled grid loops under a dataflow region
    assert "#pragma HLS dataflow" in cpp
    assert "#pragma HLS unroll" in cpp
    assert "void top(" in cpp
    # the four roles plus top are each defined exactly once
    for name in ("pe_interior", "load_a", "load_b", "drain", "top"):
        assert cpp.count(f"void {name}(") == 1
    # no per-grid-point function names leaked in
    assert not re.search(r"void \w+_\d+_\d+\(", cpp)


def test_rolled_body_count_is_constant_across_grid_sizes():
    small = emit_rolled_hls(_systolic_twin(4, 4, 4))
    large = emit_rolled_hls(_systolic_twin(8, 8, 4))
    # the synthesizer sees the same distinct function bodies at 4x4 and 8x8 -- O(#roles),
    # not O(P0*P1). Only the compile-time grid extents (the #define lines) differ.
    assert _role_bodies(small) == _role_bodies(large)
    assert len(_role_bodies(small)) == 5
    assert "#define M 4" in small and "#define M 8" in large


def test_rolled_build_target_emits_csynth_ready_project():
    # the rolled O(#roles) synthesis path is reachable through the public build API
    with tempfile.TemporaryDirectory() as tmp:
        project = spmw.build(_systolic_twin(4, 4, 4), target="rolled", project=tmp)
        assert project.hls_code == emit_rolled_hls(_systolic_twin(4, 4, 4))
        kernel = os.path.join(tmp, "kernel.cpp")
        tcl = os.path.join(tmp, "run.tcl")
        assert os.path.exists(kernel) and os.path.exists(tcl)
        with open(tcl, encoding="utf-8") as handle:
            script = handle.read()
        # the script csynths the single rolled top, not one project per grid point
        assert "csynth_design" in script
        assert "set_top top" in script
        assert "add_files kernel.cpp" in script
