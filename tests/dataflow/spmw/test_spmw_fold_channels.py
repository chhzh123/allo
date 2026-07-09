# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import allo.spmw as spmw
from allo.ir.types import float32


def _twin(M, N, K, **map_kwargs):
    """A systolic GEMM twin whose spmw.map forwards fold=/unroll= (or nothing)."""
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
        spmw.map(pe, grid=grid, **map_kwargs)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    return gemm


def _resolve(**map_kwargs):
    module = spmw.lower(_twin(4, 4, 4, **map_kwargs))
    spmw._run_module_pass(module, "spmw-resolve-channels")
    return str(module)


def _families(ir, attr):
    match = re.search(re.escape(attr) + r" = \[([^\]]*)\]", ir)
    return re.findall(r'"([^"]+)"', match.group(1)) if match else []


def test_unfolded_mesh_keeps_both_families_as_streams():
    # an unfolded mesh is byte-identical to before: both families are FIFO-backed, no buffers
    ir = _resolve()
    assert sorted(_families(ir, "spmw.channel_families")) == [
        "east/west",
        "north/south",
    ]
    assert "spmw.buffer_families" not in ir


def test_fold_along_connection_axis_reclassifies_that_family_to_buffer():
    # folding the column axis breaks the east/west forwarding chain's FIFO order -> buffer;
    # north/south (a row channel) is untouched and stays a stream
    ir = _resolve(fold={"col": 2})
    assert _families(ir, "spmw.channel_families") == ["north/south"]
    assert _families(ir, "spmw.buffer_families") == ["east/west"]


def test_fold_along_row_axis_reclassifies_north_south():
    ir = _resolve(fold={"row": 2})
    # north/south spans the row axis -> buffer; east/west (a column channel) stays a stream
    assert _families(ir, "spmw.channel_families") == ["east/west"]
    assert _families(ir, "spmw.buffer_families") == ["north/south"]


def test_fold_along_non_connection_axis_keeps_family_a_stream():
    # the east/west family spans only the column axis, so folding rows does not reclassify it
    ir = _resolve(fold={"row": 2})
    assert "east/west" in _families(ir, "spmw.channel_families")
    assert "east/west" not in _families(ir, "spmw.buffer_families")


def test_fold_factor_one_is_not_a_fold():
    # an explicit fold of 1 on every axis time-multiplexes nothing, so both families stream
    ir = _resolve(fold=[1, 1])
    assert sorted(_families(ir, "spmw.channel_families")) == [
        "east/west",
        "north/south",
    ]
    assert "spmw.buffer_families" not in ir
