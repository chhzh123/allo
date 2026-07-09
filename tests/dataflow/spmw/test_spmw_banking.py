# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""XOR/F2 partition-function banking + static injectivity verification (task4.3, AC-6).

A banked buffer may carry a *partition function* -- how many banks it splits into and which bank a
logical index lives in. The function is statically verified injective over each cycle's access set
(the ``banks`` indices a fold/unroll makes simultaneous): a conflict-free function lowers to a
``spmw.bank_functions`` entry and, in HLS, a ``cyclic factor=banks`` partition; a colliding function
is rejected (or, on request, serialized to one bank with a reported warning) -- never silently wrong.
"""

import re
import pytest
import allo.spmw as spmw
from allo.spmw_hls import emit_rolled_hls_ir
from allo.ir.types import float32


# ---------------------------------------------------------------------------------------------------
# Frontend surface: banked(...) carries a partition function and validates its arguments.
# ---------------------------------------------------------------------------------------------------


def test_banked_carries_partition_function():
    buf = spmw.banked(float32[8, 8], on="col", banks=4, bank="xor")
    assert buf.kind == "banked"
    assert buf.bank_axis == 1  # on="col"
    assert buf.banks == 4
    assert buf.bank == "xor"
    assert buf.on_conflict == "error"


def test_plain_banked_has_no_partition_function():
    # no bank=/banks= -> a fully partitioned banked buffer, unchanged from before
    buf = spmw.banked(float32[8, 8], on="row")
    assert buf.bank is None and buf.banks is None


def test_bank_without_banks_rejected():
    with pytest.raises(spmw.SPMWError, match="needs banks="):
        spmw.banked(float32[8, 8], on="row", bank="cyclic")


def test_bad_on_conflict_rejected():
    with pytest.raises(spmw.SPMWError, match="on_conflict"):
        spmw.banked(
            float32[8, 8], on="row", banks=4, bank="cyclic", on_conflict="ignore"
        )


def test_non_positive_banks_rejected():
    with pytest.raises(spmw.SPMWError, match="banks must be positive"):
        spmw.banked(float32[8, 8], on="row", banks=0, bank="cyclic")


def test_bad_bank_type_rejected():
    with pytest.raises(spmw.SPMWError, match="bank= must be"):
        spmw.banked(float32[8, 8], on="row", banks=4, bank=3.5)


# ---------------------------------------------------------------------------------------------------
# Lowering: a verified partition function rides on the rolled IR as spmw.bank_functions.
# ---------------------------------------------------------------------------------------------------


def _region_with_banking(buffer, shape=(8, 8)):
    grid = spmw.mesh((1, 4))

    @spmw.unit
    def pe(ctx):
        pass

    @spmw.region()
    def top(C: float32[shape[0], shape[1]]):
        spmw.map(pe, grid=grid)
        spmw.place("C", buffer)

    return top


def _bank_entry(ir, tensor):
    match = re.search(r"spmw\.bank_functions = \[([^\]]*)\]", ir, re.DOTALL)
    if not match:
        return None
    for entry in re.findall(r'"([^"]+)"', match.group(1)):
        if entry.split(":")[0] == tensor:
            return entry
    return None


def test_cyclic_banking_lowers_to_bank_functions():
    # cyclic on the row axis (size 8), banks=4 -> bank(i) = i % 4
    buf = spmw.banked(float32[8, 8], on="row", banks=4, bank="cyclic")
    ir = str(spmw.lower(_region_with_banking(buf)))
    assert _bank_entry(ir, "C") == "C:4:0,1,2,3,0,1,2,3"


def test_xor_banking_is_conflict_free_and_lowers():
    # an XOR swizzle on the col axis (size 8), banks=4: (low ^ high) mod 4 -- a non-trivial map that
    # is still injective over every aligned 4-window, so it verifies and lowers.
    buf = spmw.banked(float32[8, 8], on="col", banks=4, bank="xor")
    ir = str(spmw.lower(_region_with_banking(buf)))
    assert _bank_entry(ir, "C") == "C:4:0,1,2,3,1,0,3,2"


def test_mask_banking_resolves_banks_from_mask_count():
    # a list of GF(2) masks implies banks = 2**len(masks); [1, 2] -> bank(i) = i's low two bits = i%4
    buf = spmw.banked(float32[8, 8], on="row", bank=[1, 2])
    ir = str(spmw.lower(_region_with_banking(buf)))
    assert _bank_entry(ir, "C") == "C:4:0,1,2,3,0,1,2,3"


def test_callable_bank_function_lowers():
    # a non-affine Python bank function, injective over each 4-window
    buf = spmw.banked(float32[8, 8], on="row", banks=4, bank=lambda i: (i + 1) % 4)
    ir = str(spmw.lower(_region_with_banking(buf)))
    assert _bank_entry(ir, "C") == "C:4:1,2,3,0,1,2,3,0"


def test_unbanked_placement_emits_no_bank_functions():
    # a plain banked buffer (no partition function) -> no spmw.bank_functions attr (byte-identical)
    buf = spmw.banked(float32[8, 8], on="row")
    ir = str(spmw.lower(_region_with_banking(buf)))
    assert "spmw.bank_functions" not in ir


# ---------------------------------------------------------------------------------------------------
# Static injectivity: a function that collides over a cycle's access set is rejected -- or serialized.
# ---------------------------------------------------------------------------------------------------


def test_non_injective_bank_rejected():
    # two indices in the same aligned 4-window map to the same bank -> a conflict, rejected
    buf = spmw.banked(float32[8, 8], on="row", banks=4, bank=lambda i: (i // 2) % 4)
    with pytest.raises(spmw.SPMWError, match="not injective over a cycle's access set"):
        spmw.lower(_region_with_banking(buf))


def test_out_of_range_bank_rejected():
    # a bank index outside [0, banks) is a bug in the function itself -> rejected even under serialize
    buf = spmw.banked(
        float32[8, 8], on="row", banks=4, bank=lambda i: 9, on_conflict="serialize"
    )
    with pytest.raises(spmw.SPMWError, match=r"outside \[0, 4\)"):
        spmw.lower(_region_with_banking(buf))


def test_serialize_fallback_warns_and_marks_ir():
    # on_conflict="serialize" turns a conflict into a single-bank fallback with a reported warning
    buf = spmw.banked(
        float32[8, 8], on="row", banks=4, bank=lambda i: 0, on_conflict="serialize"
    )
    with pytest.warns(UserWarning, match="serializing to a single bank"):
        ir = str(spmw.lower(_region_with_banking(buf)))
    entry = _bank_entry(ir, "C")
    assert entry is not None and entry.endswith(":serialized")
    assert entry.startswith("C:1:")  # collapsed to one bank


# ---------------------------------------------------------------------------------------------------
# HLS emission: banked buffers become cyclic partitions; a non-cyclic swizzle records its bank map.
# ---------------------------------------------------------------------------------------------------


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


def test_cyclic_banking_emits_cyclic_partition():
    twin = _twin(
        4,
        4,
        4,
        placements=[
            (
                "C",
                spmw.banked(
                    float32[4, 4], on="row", banks=2, bank="cyclic", resource="URAM"
                ),
            )
        ],
    )
    cpp = emit_rolled_hls_ir(twin)
    # two physical banks along the row axis (tensor axis 0 -> HLS dim 1), pinned to URAM
    assert "#pragma HLS array_partition variable=C cyclic factor=2 dim=1" in cpp
    assert "#pragma HLS bind_storage variable=C type=RAM_2P impl=URAM" in cpp
    # a plain cyclic map needs no explicit swizzle table
    assert "conflict-free F2/XOR banking for C" not in cpp


def test_xor_banking_emits_partition_and_swizzle_comment():
    twin = _twin(
        4,
        4,
        4,
        placements=[("C", spmw.banked(float32[4, 4], on="row", banks=2, bank="xor"))],
    )
    cpp = emit_rolled_hls_ir(twin)
    assert "#pragma HLS array_partition variable=C cyclic factor=2 dim=1" in cpp
    # the non-cyclic swizzle records its resolved index-to-bank map ([0, 1, 1, 0] for size 4/banks 2)
    assert (
        "conflict-free F2/XOR banking for C (banks=2): bank index by logical index = [0, 1, 1, 0]"
        in cpp
    )


def test_serialized_banking_emits_single_bank_note():
    twin = _twin(
        4,
        4,
        4,
        placements=[
            (
                "C",
                spmw.banked(
                    float32[4, 4],
                    on="row",
                    banks=2,
                    bank=lambda i: 0,
                    on_conflict="serialize",
                ),
            )
        ],
    )
    with pytest.warns(UserWarning, match="serializing to a single bank"):
        cpp = emit_rolled_hls_ir(twin)
    assert "// banking for C serialized to one bank (bank conflict reported)" in cpp
    assert "#pragma HLS array_partition variable=C complete dim=0" in cpp


def test_unbanked_operands_keep_default_partition():
    # a partition-function placement on C must not disturb the unplaced A/B defaults
    twin = _twin(
        4,
        4,
        4,
        placements=[("C", spmw.banked(float32[4, 4], on="row", banks=2, bank="xor"))],
    )
    cpp = emit_rolled_hls_ir(twin)
    assert "#pragma HLS array_partition variable=A complete dim=0" in cpp
    assert "#pragma HLS array_partition variable=B complete dim=0" in cpp
