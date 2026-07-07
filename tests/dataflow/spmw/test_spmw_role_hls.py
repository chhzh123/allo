# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo.spmw as spmw
from allo.spmw_hls import emit_role_hls
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
