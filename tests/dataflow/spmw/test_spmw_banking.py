# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""XOR-swizzled banking: does it do what its docstring claims?

`spmw.xor_bank` was exported and documented as "the conflict-free layout for
butterfly access sets" while doing nothing at all -- `Brick.layout` was assigned
and never read anywhere in the repository. These tests are the claim itself,
checked: a butterfly's two operands land in different banks, and the mapping
loses nothing.

The formula is the one the vectorised FFT-256 reference uses::

    bank(i) = (i & (W - 1)) ^ (((i >> s) & 1) << (log2(W) - 1))
"""

import pytest

import allo.spmw as spmw
from allo.spmw.bricks import Layout
from allo.spmw.errors import SPMWMemoryError

WIDTH = 32  # the reference's lane count
LOG2_W = 5
N = 256


def _pairs(stage, n=N):
    """The index pairs a butterfly reads at this stage."""
    span = 1 << stage
    for block in range(0, n, 2 * span):
        for offset in range(span):
            yield block + offset, block + offset + span


def test_a_bank_index_is_a_bank_index():
    """Whatever the swizzle does, it has to land in range."""
    layout = spmw.xor_bank(WIDTH, stride_bit=LOG2_W)
    seen = set()
    for index in range(N):
        bank = layout.bank_of(index)
        assert 0 <= bank < WIDTH
        seen.add(bank)
    assert seen == set(range(WIDTH)), "every bank should be reachable"


@pytest.mark.parametrize("stage", [LOG2_W, LOG2_W + 1, LOG2_W + 2])
def test_a_butterflys_operands_land_in_different_banks(stage):
    """The whole point, at the stages where it matters.

    Once the stride reaches the bank count the two operands share their low
    bits, so cyclic banking collides on every pair. The swizzle folds bit `s`
    into the bank index, and bit `s` is exactly what differs between them.
    """
    layout = spmw.xor_bank(WIDTH, stride_bit=stage)
    for up, lo in _pairs(stage):
        assert layout.bank_of(up) != layout.bank_of(lo), (stage, up, lo)


@pytest.mark.parametrize("stage", [LOG2_W, LOG2_W + 1, LOG2_W + 2])
def test_cyclic_banking_collides_on_every_pair(stage):
    """The control, so the test above is measuring the swizzle and not luck.

    If plain banking happened to work there would be nothing to swizzle, and
    the previous test would pass for reasons that had nothing to do with XOR.
    """
    plain = spmw.banked(banks=WIDTH)
    collisions = sum(plain.bank_of(up) == plain.bank_of(lo) for up, lo in _pairs(stage))
    assert collisions == len(list(_pairs(stage))), (
        "cyclic banking should collide on *every* pair once the stride reaches "
        "the bank count; if it does not, this stage is not the conflicting case"
    )


def test_below_the_bank_count_there_is_nothing_to_fix():
    """Stages whose stride is inside a bank do not conflict to begin with.

    The reference swizzles only stages 5..7 for a reason: at stride < WIDTH the
    two operands already differ in their low bits.
    """
    plain = spmw.banked(banks=WIDTH)
    for stage in range(LOG2_W):
        for up, lo in _pairs(stage):
            assert plain.bank_of(up) != plain.bank_of(lo), (stage, up, lo)


@pytest.mark.parametrize("stage", [LOG2_W, LOG2_W + 1, LOG2_W + 2])
def test_the_swizzle_loses_nothing(stage):
    """(bank, row) must be a bijection, or the layout destroys data.

    Separating the operands is easy if you are allowed to map two indices onto
    one slot. This is the property that makes the swizzle a *layout* rather
    than a hash, and no docstring in the original mentioned it.
    """
    layout = spmw.xor_bank(WIDTH, stride_bit=stage)
    slots = [layout.place(index) for index in range(N)]
    assert len(set(slots)) == N, "two indices share a slot"
    rows = N // WIDTH
    assert set(slots) == {(b, r) for b in range(WIDTH) for r in range(rows)}


def test_a_swizzle_inside_the_bank_bits_is_refused():
    """Why `stride_bit` has to be at or above log2(banks).

    With s below log2(banks) the swizzle XORs a bank bit against itself, which
    pins it to zero and folds two indices onto one slot -- and there was no
    conflict to fix there in the first place. Silently losing half a buffer is
    the worst way for this to go wrong, so it raises.
    """
    with pytest.raises(SPMWMemoryError, match="inside the bank index"):
        spmw.xor_bank(WIDTH, stride_bit=LOG2_W - 1)
    with pytest.raises(SPMWMemoryError, match="inside the bank index"):
        spmw.xor_bank(WIDTH).at_stride(0)


def test_a_bank_count_must_be_a_power_of_two():
    """The swizzle is defined on the low log2(banks) bits."""
    with pytest.raises(SPMWMemoryError, match="power-of-two"):
        spmw.xor_bank(24, stride_bit=LOG2_W)


def test_at_stride_specialises_without_mutating():
    """One declared layout, one specialisation per stage."""
    base = spmw.xor_bank(WIDTH)
    assert base.stride_bit is None
    fifth = base.at_stride(LOG2_W)
    assert fifth.stride_bit == LOG2_W
    assert base.stride_bit is None, "at_stride should not mutate the original"
    assert isinstance(fifth, Layout)
    assert fifth.banks == base.banks and fifth.bank_fn == base.bank_fn
