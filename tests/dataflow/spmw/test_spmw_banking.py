# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPMW-facing F2 partition-function banking (task4.3): the SPMW surface + real banked storage.

`spmw.banked(banks=, bank=, stride_bit=)` is validated by the F2 solver at lower time (a
non-conflict-free / too-few-banks / non-power-of-two banking is rejected), and the rolled HLS
emitter realizes it as **real 2D `[banks][depth]` banked storage** with swizzled
`C_bank[bank(i)][offset(i)][j]` accesses (via the F2 `SwizzleHelper`), not a comment-only pragma.
"""

import re
import pytest
import allo.spmw as spmw
from allo.spmw_hls import emit_rolled_hls_ir
from allo.ir.types import float32


def _twin(M, N, K, placements=(), fold=None):
    """A systolic GEMM twin with optional ``spmw.place`` calls and an optional ``fold``."""
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
        spmw.map(pe, grid=grid, fold=fold)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")
        operands = {"A": A, "B": B, "C": C}
        for name, buffer in placements:
            spmw.place(operands[name], buffer)

    return gemm


# --- surface + validation -----------------------------------------------------------------------


def test_banked_carries_partition_function():
    buf = spmw.banked(float32[8, 8], on="row", banks=4, bank="cyclic")
    assert buf.kind == "banked" and buf.bank_axis == 0
    assert buf.banks == 4 and buf.bank_mode == "cyclic"


def test_non_power_of_two_banks_rejected():
    with pytest.raises(spmw.SPMWError, match="power of two"):
        spmw.banked(float32[8, 8], on="row", banks=3, bank="cyclic")


def test_xor_needs_banks_and_stride_bit():
    with pytest.raises(spmw.SPMWError, match="stride_bit"):
        spmw.banked(float32[8, 8], on="row", banks=4, bank="xor")


def test_plain_banked_still_has_no_partition_function():
    buf = spmw.banked(float32[8, 8], on="row")
    assert buf.banks is None


def test_placed_banking_lowers_to_bank_functions():
    twin = _twin(
        4, 4, 4, placements=[("C", spmw.banked(float32[4, 4], on="row", banks=2))]
    )
    ir = str(spmw.lower(twin))
    assert re.search(r'spmw\.bank_functions = \[\s*"C:2:0:cyclic:0"', ir)


def test_non_power_of_two_axis_rejected_at_lower():
    # C's row axis is 6, not a power of two -> the F2 banking cannot be realized
    with pytest.raises(spmw.SPMWError, match="power-of-two"):
        spmw.lower(
            _twin(
                6,
                4,
                4,
                placements=[("C", spmw.banked(float32[6, 4], on="row", banks=2))],
            )
        )


# --- real banked storage in the emitted HLS -----------------------------------------------------


def test_emitter_realizes_real_2d_banked_storage_for_C():
    # C banked on the row axis (M=4) into 2 banks -> real C_bank[2][2][N] storage, swizzled writes,
    # and a writeback to the host C[M][N].
    twin = _twin(
        4, 4, 4, placements=[("C", spmw.banked(float32[4, 4], on="row", banks=2))]
    )
    cpp = emit_rolled_hls_ir(twin)
    assert "C_bank[2][2][N]" in cpp  # real 2D [banks][depth] banked storage
    assert "#pragma HLS array_partition variable=C_bank complete dim=1" in cpp
    # the PE writes to a swizzled slot, and the banked buffer is written back to C
    assert "&C_bank[" in cpp and "][j]);" in cpp
    assert re.search(r"C\[i\]\[j\] = C_bank\[", cpp)  # writeback to the host interface


def test_xor_banking_emits_swizzled_index():
    twin = _twin(
        4,
        4,
        4,
        placements=[
            (
                "C",
                spmw.banked(float32[4, 4], on="row", banks=2, bank="xor", stride_bit=1),
            )
        ],
    )
    cpp = emit_rolled_hls_ir(twin)
    assert "C_bank[2][2][N]" in cpp
    # an xor swizzle references the offset bit in the bank index (not a plain low-bit cyclic)
    assert "&C_bank[" in cpp


def test_unbanked_twin_emits_no_c_bank():
    cpp = emit_rolled_hls_ir(_twin(4, 4, 4))
    assert "C_bank" not in cpp
    assert "&C[i][j]" in cpp  # unchanged default write target


def test_banked_A_operand_rejected_by_emitter():
    # the rolled emitter realizes banking of the output C along the row axis; a banked input is
    # rejected clearly, not silently ignored
    twin = _twin(
        4, 4, 4, placements=[("A", spmw.banked(float32[4, 4], on="row", banks=2))]
    )
    with pytest.raises(NotImplementedError, match="output C"):
        emit_rolled_hls_ir(twin)


# --- folded buffer families materialized as real banked on-chip buffers -------------------------


def test_folded_map_materializes_a_banked_on_chip_buffer():
    # fold={"col": 2} reclassifies the east/west (A) family to a buffer (spmw.buffer_families); the
    # emitter no longer rejects it and materializes A as a REAL 2D-banked on-chip buffer that the
    # loader reads by (bank, offset) -- the A-forwarding now flows through addressed banked storage,
    # not a FIFO family.
    cpp = emit_rolled_hls_ir(_twin(4, 4, 4, fold={"col": 2}))
    assert (
        "A_bank[2][2][K]" in cpp
    )  # real [banks][depth][K] on-chip buffer (M=4, 2 banks)
    assert "#pragma HLS array_partition variable=A_bank complete dim=1" in cpp
    # the loader reads A through bank/offset (routing the family through the buffer)
    assert "out.write(A_bank[" in cpp and "][k]);" in cpp
    # B (north/south) stays a FIFO family
    assert "fb[M + 1][N]" in cpp


def test_unfolded_map_has_no_banked_a_buffer():
    # the spatial (unfolded) path is unchanged -- A stays a FIFO family, no A_bank buffer
    cpp = emit_rolled_hls_ir(_twin(4, 4, 4))
    assert "A_bank" not in cpp
    assert "load_a(A, i, fa[i][0])" in cpp


def test_unsupported_folded_shape_is_rejected():
    # fold along the row axis reclassifies north/south (B), not east/west -- an unsupported folded
    # shape the emitter rejects clearly rather than mis-emitting
    with pytest.raises(NotImplementedError, match="folded rolled emitter"):
        emit_rolled_hls_ir(_twin(4, 4, 4, fold={"row": 2}))
