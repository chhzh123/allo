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

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import float32
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


# -- the layout as the lowering sees it --------------------------------------

BANKS, ROWS = 8, 4
SIZE = BANKS * ROWS


class _ProbeIO(spmw.Interface):
    x_in = spmw.In(float32)
    y_out = spmw.Out(float32)
    tab = spmw.MemIn(float32[SIZE])


@spmw.unit
def _probe(io: _ProbeIO, site: spmw.Site):
    (k,) = site.rank
    v = io.x_in.get()
    io.y_out.put(v + io.tab[k])


def _fabric_with(layout):
    grid = spmw.Grid((SIZE,))

    @spmw.fabric
    def fab(X: float32[SIZE], Y: float32[SIZE]):
        P = spmw.place(_probe, on=grid)
        t = spmw.mem(
            float32[SIZE], init=np.arange(SIZE, dtype=np.float32), layout=layout
        )
        spmw.stationary(t, at=P.tab)
        (lane,) = P.axes
        spmw.stream_in(X, into=P.x_in, index=(lane,))
        spmw.gather(Y, from_=P.y_out, index=(lane,))

    return fab


def test_banking_is_invisible_to_the_answer():
    """A layout moves data; it does not change what the data is.

    The swizzle is a bijection, so a banked brick and a plain one must give the
    same result for every input. If they ever differ, the permutation and the
    emitted subscript have drifted apart -- which is the failure this layout is
    most likely to have and the hardest to see in a waveform.
    """
    X = np.arange(SIZE, dtype=np.float32) * 10
    out = {}
    for name, layout in (
        ("plain", spmw.replicate),
        ("banked", spmw.xor_bank(BANKS, stride_bit=LOG2_W - 2)),
    ):
        Y = np.zeros(SIZE, dtype=np.float32)
        spmw.build(_fabric_with(layout), target="ref")(X, Y)
        out[name] = Y.copy()
    np.testing.assert_array_equal(out["plain"], out["banked"])
    # And it did compute something, so equality is not two zero arrays.
    assert out["plain"].any()


def test_the_swizzle_reaches_the_emitted_design():
    """Banking that changes no code is the bug this whole file exists for.

    `xor_bank` was exported and documented for months while `Brick.layout` was
    never read. Equal answers alone would not catch that -- doing nothing also
    gives equal answers.
    """
    plain = spmw.source(_fabric_with(spmw.replicate))
    banked = spmw.source(_fabric_with(spmw.xor_bank(BANKS, stride_bit=LOG2_W - 2)))
    assert "^" not in plain
    assert "^" in banked and ">>" in banked


def test_a_multidimensional_brick_cannot_be_banked_silently():
    """Banking splits one linear address space; two axes is a different ask."""
    grid = spmw.Grid((SIZE,))

    class TwoD(spmw.Interface):
        x_in = spmw.In(float32)
        y_out = spmw.Out(float32)
        tab = spmw.MemIn(float32[BANKS, ROWS])

    @spmw.unit
    def unit(io: TwoD, site: spmw.Site):
        (k,) = site.rank
        io.y_out.put(io.x_in.get() + io.tab[0, 0])

    @spmw.fabric
    def fab(X: float32[SIZE], Y: float32[SIZE]):
        P = spmw.place(unit, on=grid)
        t = spmw.mem(
            float32[BANKS, ROWS],
            init=np.zeros((BANKS, ROWS), dtype=np.float32),
            layout=spmw.xor_bank(BANKS, stride_bit=LOG2_W - 2),
        )
        spmw.stationary(t, at=P.tab)
        (lane,) = P.axes
        spmw.stream_in(X, into=P.x_in, index=(lane,))
        spmw.gather(Y, from_=P.y_out, index=(lane,))

    with pytest.raises(SPMWMemoryError, match="one-dimensional"):
        spmw.source(fab)


def test_banking_without_a_stride_is_refused():
    """There is no conflict-free layout without an access pattern to be free of."""
    with pytest.raises(SPMWMemoryError, match="no stride bit"):
        spmw.source(_fabric_with(spmw.xor_bank(BANKS)))


# -- the closed form against the solver ---------------------------------------


@pytest.mark.parametrize("stage", [LOG2_W, LOG2_W + 1, LOG2_W + 2])
def test_the_closed_form_agrees_with_the_f2_solver(stage):
    """Two independent derivations of the same swizzle.

    `Layout.bank_of` is the closed form -- low bits XOR the stride's bit -- and
    is what the lowering emits. `F2LayoutSolver` builds the conflict subspace
    over GF(2) and solves for a bank-selection matrix. They should agree
    everywhere for a single stride; if they ever stop, the closed form is a
    special case that has quietly become wrong.
    """
    layout = spmw.xor_bank(WIDTH, stride_bit=stage)
    helper = layout.solved(N)
    for index in range(N):
        assert layout.bank_of(index) == helper.swizzle_bank(index), index
        assert layout.row_of(index) == helper.bank_offset(index), index


def test_the_solver_and_the_layout_agree_on_shape():
    """`dims()` is (banks, rows), which is what the brick is reshaped to."""
    layout = spmw.xor_bank(WIDTH, stride_bit=LOG2_W)
    assert layout.solved(N).dims() == (WIDTH, N // WIDTH)


def test_the_solver_refuses_a_stride_inside_the_bank_bits_too():
    """The constraint is not something SPMW invented locally.

    `fft_swizzle` asserts `stride_bit >= bank_bits` for the same reason
    `Layout.at_stride` raises: below that there is no conflict to fix and the
    swizzle would fold two indices onto one slot.
    """
    from allo.transform.f2_layout import fft_swizzle

    with pytest.raises(AssertionError, match="must be >="):
        fft_swizzle(N, WIDTH, LOG2_W - 1)
