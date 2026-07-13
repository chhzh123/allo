# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import pytest
import allo.spmw as spmw


def _mesh_region(shape):
    grid = spmw.mesh(shape)

    @spmw.unit
    def pe(ctx):
        a = ctx.west.get()
        b = ctx.north.get()
        ctx.east.put(a)
        ctx.south.put(b)

    @spmw.region()
    def gemm(A):
        spmw.map(pe, grid=grid)

    return gemm


def test_role_body_count_is_nine_for_any_mesh_size():
    # the hard metric: the synthesized function-body count is bounded by the role count
    for shape in [(8, 8), (16, 16), (32, 32)]:
        assert spmw.role_body_count(_mesh_region(shape)) == 9


def test_emitted_hls_body_count_is_constant_not_p_squared():
    # 8x8 and 32x32 emit the SAME nine role bodies -- not P0*P1 cloned bodies
    assert spmw.emit_hls(_mesh_region((8, 8))).count("void gemm_pe_") == 9
    assert spmw.emit_hls(_mesh_region((32, 32))).count("void gemm_pe_") == 9


def test_emitted_hls_has_rolled_dataflow_structure():
    cpp = spmw.emit_hls(_mesh_region((8, 8)))
    assert "#pragma HLS dataflow" in cpp
    assert "#pragma HLS unroll" in cpp  # rolled instantiation loop
    assert "hls::stream<float> fifo_" in cpp  # per-key FIFO arrays


def test_no_per_pid_cloned_body_names():
    cpp = spmw.emit_hls(_mesh_region((8, 8)))
    assert re.search(r"gemm_pe_\d+_\d+", cpp) is None
    assert re.search(r"gemm_\d+_\d+", cpp) is None


def test_degenerate_grid_yields_smaller_body_count():
    assert spmw.role_body_count(_mesh_region((2, 4))) == 6
    assert spmw.role_body_count(_mesh_region((2, 2))) == 4


def test_emit_hls_rejects_non_mesh():
    grid = spmw.ring(4)  # ports next/prev, not a nearest-neighbor mesh

    @spmw.unit
    def pe(ctx):
        ctx.next.get()

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid)

    with pytest.raises(spmw.SPMWError, match="mesh"):
        spmw.emit_hls(r)


def test_rolled_hls_translates_integer_datapath():
    """The IR-based rolled-HLS emitter translates an integer systolic MAC PE: `spmw_mlir` emits
    arith.muli/addi for an int datapath, and the IR->C++ translator now recognizes the integer
    mnemonics (previously only float ones, so integer regions raised NotImplementedError despite
    int32/int64 being in the dtype table)."""
    from allo.ir.types import int32
    from allo.spmw_hls import emit_rolled_hls_ir

    grid = spmw.mesh((3, 3))

    @spmw.unit
    def pe(ctx):
        c: int32 = 0
        for k in range(2):
            a: int32 = ctx.west.get()
            b: int32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: int32[3, 2], B: int32[2, 3], C: int32[3, 3]):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    cpp = emit_rolled_hls_ir(gemm)  # no NotImplementedError on the integer arith
    assert "int c_local[1]" in cpp  # emitted as an int (not float) datapath
    assert (
        "c_local[0] =" in cpp
    )  # the integer MAC accumulation reaches the output store
