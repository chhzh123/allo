# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""F2 conflict-free bank partitioning (task4.3): the real banking machinery.

`allo.transform.f2_layout` maps a 1D buffer to a 2D ``[num_banks, depth]`` banked layout with an
F2 (GF(2)) swizzle so that the addresses a butterfly stage accesses *in parallel* -- the pair
``{i, i ^ (1 << s)}`` for stride bit ``s`` -- always land in **distinct banks** (conflict-free at
II=1). Injectivity is not checked over synthetic windows: `F2LayoutSolver` builds the conflict
subspace from the parallel-access strides and solves for a swizzle matrix that separates them, and
the (bank, offset) map is a lossless bijection over the whole address space.
"""

import tempfile

import allo
from allo.ir.types import int32
from allo.transform.f2_layout import F2LayoutSolver, SwizzleHelper, fft_swizzle


def test_butterfly_access_is_conflict_free_across_all_stages():
    # 256-point transform, 2 banks: at every butterfly stride bit s, the pair {i, i^(1<<s)} is
    # accessed together and must fall in different banks.
    n_bits, bank_bits = 8, 1
    for s in range(n_bits):
        helper = F2LayoutSolver(n_bits, bank_bits).solve([s])
        for i in range(1 << n_bits):
            j = i ^ (1 << s)
            assert helper.swizzle_bank(i) != helper.swizzle_bank(
                j
            ), f"stage {s}: indices {i},{j} collide in bank {helper.swizzle_bank(i)}"


def test_multi_bank_butterfly_separates_parallel_strides():
    # 4 banks separating two simultaneous butterfly strides {0, 1}: every group of indices that
    # differ only in bits 0/1 must occupy 4 distinct banks.
    helper = F2LayoutSolver(8, 2).solve([0, 1])
    for base in range(0, 1 << 8, 4):
        banks = {helper.swizzle_bank(base ^ d) for d in range(4)}
        assert len(banks) == 4, f"base {base}: banks {banks} not all distinct"


def test_swizzle_is_a_lossless_bijection():
    # Every address maps to a unique (bank, offset) slot -- the 1D->2D layout loses nothing.
    helper = F2LayoutSolver(8, 2).solve([0, 1])
    slots = set()
    for i in range(1 << 8):
        slot = (helper.swizzle_bank(i), helper.bank_offset(i))
        assert slot not in slots, f"address {i} collides at slot {slot}"
        slots.add(slot)
    assert len(slots) == (1 << 8)
    num_banks, depth = helper.dims()
    assert (num_banks, depth) == (4, 64)


def test_bank_and_offset_exprs_are_valid_cpp():
    # The helper emits C++ index expressions the HLS backend inlines at each banked access.
    helper = F2LayoutSolver(8, 1).solve([3])
    bank = helper.bank_expr("idx")
    offset = helper.offset_expr("idx")
    assert "idx" in bank
    assert offset == "(idx >> 1)"


def test_fft_swizzle_is_conflict_free_for_its_stage():
    # fft_swizzle is the convenience wrapper the FFT datapath uses. WIDTH=8 -> 8 banks
    # (bank_bits=3); an inter-vector butterfly stage (stride_bit >= bank_bits) must be conflict-free.
    stride_bit = 4
    helper = fft_swizzle(256, 8, stride_bit)
    assert isinstance(helper, SwizzleHelper)
    for i in range(256):
        j = i ^ (1 << stride_bit)
        assert helper.swizzle_bank(i) != helper.swizzle_bank(j)


def test_bind_storage_and_dependence_pragmas_emit_in_hls():
    # The banking machinery pins a buffer's storage resource and marks it dependence-free; these
    # rode in on the ported EmitVivadoHLS.cpp hunks (emitAlloc). Exercise them directly (the
    # f2_layout 1D->2D rewrite itself is exercised by the FFT butterfly kernels in task4.4).
    def banked_copy(inp: int32[16]) -> int32[16]:
        buf: int32[16] = 0
        for i in range(16):
            buf[i] = inp[i] + 1
        out: int32[16] = 0
        for i in range(16):
            out[i] = buf[i]
        return out

    s = allo.customize(banked_copy)
    s.bind_storage("banked_copy:buf", impl="uram", storage_type="ram_2p")
    s.dependence("banked_copy:buf")
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
    code = mod.hls_code
    assert "#pragma HLS bind_storage" in code and "impl=uram" in code
    assert "type=ram_2p" in code
    assert "#pragma HLS dependence" in code and "inter false" in code
