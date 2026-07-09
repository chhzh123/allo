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
import pytest

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


def test_multiple_conflicting_strides_are_conflict_free_over_the_full_span():
    # Regression for the conflict-span bug: two conflicting strides [2, 3] make the *whole* span
    # {e2, e3, e2^e3} collide-prone. A single shared bank row let e2^e3 cancel, mapping
    # {0, 4, 8, 12} to only two banks. Each simultaneously-accessed group must now be fully distinct.
    for bank_bits in (2, 3):  # 4 and 8 banks both suffice for 2 conflicting strides
        helper = F2LayoutSolver(4, bank_bits).solve([2, 3])
        banks = [helper.swizzle_bank(a) for a in (0, 4, 8, 12)]
        assert len(set(banks)) == 4, f"bank_bits={bank_bits}: {banks} collide"


def test_three_conflicting_strides_need_three_bank_bits():
    # Three simultaneous strides span an 8-address access set; 8 banks separate them, 4 cannot.
    helper = F2LayoutSolver(6, 3).solve([3, 4, 5])
    banks = [helper.swizzle_bank(a) for a in range(0, 64, 8)]
    assert len(set(banks)) == 8


def test_solver_rejects_when_banks_too_few():
    # Too few banks to separate the simultaneous access set -> reject with a diagnostic, never a
    # silently-colliding banking.
    with pytest.raises(ValueError, match="conflict-free"):
        F2LayoutSolver(4, 1).solve(
            [2, 3]
        )  # 2 conflicting strides need 4 banks, only 2 given
    with pytest.raises(ValueError, match="conflict-free"):
        F2LayoutSolver(6, 2).solve(
            [3, 4, 5]
        )  # 3 conflicting strides need 8 banks, only 4 given


def test_single_stride_swizzle_is_unchanged():
    # The single-stride case (one FFT butterfly stage) still toggles the top bank row, matching the
    # kernel the FFT expects.
    helper = F2LayoutSolver(8, 3).solve([5])
    for i in range(1 << 8):
        assert helper.swizzle_bank(i) != helper.swizzle_bank(i ^ (1 << 5))


def test_fft_swizzle_is_conflict_free_for_its_stage():
    # fft_swizzle is the convenience wrapper the FFT datapath uses. WIDTH=8 -> 8 banks
    # (bank_bits=3); an inter-vector butterfly stage (stride_bit >= bank_bits) must be conflict-free.
    stride_bit = 4
    helper = fft_swizzle(256, 8, stride_bit)
    assert isinstance(helper, SwizzleHelper)
    for i in range(256):
        j = i ^ (1 << stride_bit)
        assert helper.swizzle_bank(i) != helper.swizzle_bank(j)


def test_f2_layout_rewrites_1d_buffer_to_real_2d_banked_storage():
    # End-to-end: Schedule.f2_layout rewrites a 1D local buffer into a real 2D [num_banks][depth]
    # banked array (not a comment) and stamps the array_partition / bind_storage / dependence
    # pragmas the banking needs. 16 elements, 2 banks -> buf[2][8].
    def bank16(inp: int32[16]) -> int32[16]:
        buf: int32[
            16
        ]  # no init: the buffer's only uses are the loads/stores f2_layout remaps
        for i in range(16):
            buf[i] = inp[i]
        out: int32[16]
        for i in range(16):
            out[i] = buf[i]
        return out

    s = allo.customize(bank16)
    s.f2_layout("bank16:buf", n_bits=4, bank_bits=1, banking="block")
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
    code = mod.hls_code
    # real 2D banked storage, banked axis fully partitioned, resource pinned, dependence relaxed
    assert "[2][8]" in code, code
    assert "#pragma HLS array_partition" in code and "complete dim=1" in code
    assert "#pragma HLS bind_storage" in code and "impl=lutram" in code.lower()
    assert "#pragma HLS dependence" in code and "inter false" in code


def test_f2_layout_fails_closed_on_unremappable_access():
    # A buffer accessed in a way the rewrite can't remap (here an init fill) must be rejected with a
    # clear error, not silently left unbanked (and never a hard MLIR abort).
    def bad(inp: int32[16]) -> int32[16]:
        buf: int32[16] = 0  # the `= 0` init fill is a use f2_layout cannot remap
        for i in range(16):
            buf[i] = inp[i]
        out: int32[16] = 0
        for i in range(16):
            out[i] = buf[i]
        return out

    s = allo.customize(bad)
    with pytest.raises(ValueError, match="cannot rewrite"):
        s.f2_layout("bad:buf", n_bits=4, bank_bits=1, banking="block")


def test_single_swizzle_cannot_separate_two_high_conflict_directions():
    # The auto_f2 fail-closed guard: a single XOR-swizzle banking separates only one high-stride
    # conflict direction, so a conflict subspace with two (e2, e3) is detected as not-conflict-free
    # -- the signal for auto_f2 to reject rather than bank silently.
    import numpy as np
    from allo.transform.auto_f2 import (
        _realized_bank_matrix,
        _banking_separates_conflicts,
    )

    P = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=np.int32)  # columns e2, e3
    for bank_bits in (1, 2):
        S = _realized_bank_matrix(
            "cyclic", stride_bit=2, bank_bits=bank_bits, n_addr_bits=4
        )
        assert not _banking_separates_conflicts(S, P)
    # a single high direction IS separable by one swizzle bit
    P1 = np.array([[0], [0], [1], [0]], dtype=np.int32)  # column e2
    S1 = _realized_bank_matrix("cyclic", stride_bit=2, bank_bits=1, n_addr_bits=4)
    assert _banking_separates_conflicts(S1, P1)


def test_solve_subspace_accepts_a_conflict_basis_directly():
    import numpy as np

    # a 2-direction conflict subspace {e2, e3} over a 4-bit address
    P = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=np.int32)
    helper = F2LayoutSolver(4, 2).solve_subspace(P)  # 4 banks suffice
    banks = [helper.swizzle_bank(a) for a in (0, 4, 8, 12)]
    assert len(set(banks)) == 4
    # too few banks -> reject
    with pytest.raises(ValueError):
        F2LayoutSolver(4, 1).solve_subspace(P)
    # a multi-bit (non-unit) conflict delta e0^e2 is not yet realizable -> reject, not silent wrong
    Q = np.array([[1], [0], [1], [0]], dtype=np.int32)
    with pytest.raises(ValueError, match="unit-stride"):
        F2LayoutSolver(4, 2).solve_subspace(Q)


def test_auto_f2_runs_end_to_end_on_a_kernel():
    # Schedule.auto_f2 analyzes a kernel's 1D buffers over F2 and applies conflict-free layouts (or
    # none when there are no conflicts); it must run end-to-end and produce buildable HLS.
    def ak(inp: int32[16]) -> int32[16]:
        buf: int32[16]  # no init: only load/store uses
        for i in range(16):
            buf[i] = inp[i]
        out: int32[16]
        for i in range(16):
            out[i] = buf[i]
        return out

    s = allo.customize(ak)
    s.auto_f2()  # no bank conflicts here -> a no-op, but must run without error
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="vitis_hls", mode="sw_emu", project=tmpdir)
    assert len(mod.hls_code) > 0 and "ak" in mod.hls_code
