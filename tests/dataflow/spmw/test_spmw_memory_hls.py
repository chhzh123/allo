# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo.spmw as spmw
from allo.spmw_hls import emit_rolled_hls_ir
from allo.ir.types import float32


def _twin(M, N, K, placements=()):
    """A systolic GEMM twin with optional ``spmw.place`` calls (each a ``(operand_name, buffer)``)."""
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
        operands = {"A": A, "B": B, "C": C}
        for name, buffer in placements:
            spmw.place(operands[name], buffer)

    return gemm


def test_placed_banked_uram_emits_bind_storage_and_bank_partition():
    twin = _twin(
        4,
        4,
        4,
        placements=[("C", spmw.banked(float32[4, 4], on="row", resource="URAM"))],
    )
    cpp = emit_rolled_hls_ir(twin)
    # C pinned to URAM and banked on the row axis (tensor axis 0 -> HLS dim 1)
    assert "#pragma HLS bind_storage variable=C type=RAM_2P impl=URAM" in cpp
    assert "#pragma HLS array_partition variable=C complete dim=1" in cpp
    # the unplaced operands keep the default full partition and get no bind_storage
    assert "#pragma HLS array_partition variable=A complete dim=0" in cpp
    assert "#pragma HLS array_partition variable=B complete dim=0" in cpp
    assert "variable=A type=RAM" not in cpp and "variable=B type=RAM" not in cpp


def test_placed_shared_bram_emits_bind_storage_with_default_partition():
    twin = _twin(
        4, 4, 4, placements=[("A", spmw.shared(float32[4, 4], resource="BRAM"))]
    )
    cpp = emit_rolled_hls_ir(twin)
    # shared (unbanked) -> a BRAM bind_storage but the default complete dim=0 partition
    assert "#pragma HLS bind_storage variable=A type=RAM_2P impl=BRAM" in cpp
    assert "#pragma HLS array_partition variable=A complete dim=0" in cpp


def test_space_level_resolves_to_resource_and_is_honored():
    # a logical space= level (no explicit resource=) still resolves to a concrete resource that the
    # emitter honors -- here L2 -> URAM
    twin = _twin(4, 4, 4, placements=[("C", spmw.shared(float32[4, 4], space="L2"))])
    cpp = emit_rolled_hls_ir(twin)
    assert "#pragma HLS bind_storage variable=C type=RAM_2P impl=URAM" in cpp


def test_auto_resource_emits_no_bind_storage():
    # a placement with neither space nor resource resolves to AUTO -> the emitter pins nothing
    twin = _twin(4, 4, 4, placements=[("C", spmw.shared(float32[4, 4]))])
    cpp = emit_rolled_hls_ir(twin)
    assert "bind_storage" not in cpp
    assert "#pragma HLS array_partition variable=C complete dim=0" in cpp


def test_unplaced_twin_emits_no_memory_pragmas():
    # no placement -> no bind_storage, every array keeps the default full partition (unchanged)
    cpp = emit_rolled_hls_ir(_twin(4, 4, 4))
    assert "bind_storage" not in cpp
    for var in ("A", "B", "C"):
        assert f"#pragma HLS array_partition variable={var} complete dim=0" in cpp
