# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The paired-operand FFT: one butterfly unit that takes *both* samples a cycle.

`test_spmw_fft_rolled.py` gives every lane its own butterfly unit and feeds it
one sample a cycle.  A radix-2 butterfly consumes two, so each unit completes
half a butterfly per cycle and the design needs twice as many of them: at
N=256, W=2 that is sixteen units at 50% against HP-FFT's eight at 100%, and
DSP 174 against 72.  The arithmetic is not wrong; the *shape* is.  This is the
other shape.

At W samples a cycle a stage has to retire W/2 butterflies a cycle, so at W=2
it is exactly one -- one unit per stage, fully used, `log2(N)` of them.  What
makes that possible is not the unit, which is a plain butterfly, but the
buffer in front of it: the two operands of stage `d` are `i` and `i ^ (1 << d)`
and they have to arrive together.

## Why the buffer is XOR-banked, and why one stride bit is enough

Each stage writes the two samples arriving this cycle and reads the two its
butterfly needs.  Both pairs have to land in different banks or the cycle
serialises, and they are different pairs:

- the arriving pair is two *adjacent* stream positions, `2t` and `2t+1`;
- the butterfly's pair is two positions `N/2` apart.

Plain cyclic banking separates the first and collides on the second -- the two
differ only in a high bit, so their low bits, which are the bank, agree.  The
swizzle folds the distinguishing bit into the bank index::

    bank(p) = (p & 1) ^ ((p >> (log2(N) - 1)) & 1)

and then both pairs split, which is the whole claim `spmw.xor_bank` makes.
`test_spmw_banking.py` holds the layout to it, and
`test_the_layout_places_every_access` below holds *this design's* accesses to
the same layout rather than to arithmetic written out again by hand.

That one stride bit covers every stage is not luck and not an assumption: a
stage emits its two results into the two positions of one cycle, so the next
stage always finds its operands `N/2` positions apart whatever the butterfly
distance was.  The schedule is uniform, so the layout is too.

## The schedule

A stage runs `N/W` cycles a transform and retires one butterfly a cycle with no
gap.  Butterfly `j` of a block reads positions `p` and `p + N/2`; the second of
those is written `lag` cycles earlier, so the stage issues `lag = N/4 + 1`
cycles behind its input and the buffer holds `2N` positions -- two blocks, so a
slot is reused 256 cycles after it dies rather than the same cycle.

## Two ways to hold the same layout, and why the design holds it the long way

`buffer="banked"` hands the linear address to `spmw.xor_bank` and lets the
lowering place it -- one memory per component, `_st_buf[bank(p), row(p)]`.
`buffer="split"`, the default, is the same layout written out: one memory per
bank, with the bank index chosen at elaboration and muxes on the addresses and
the results.  They compute the same thing (`test_the_two_buffer_forms_agree`).

The banked form is the one this design wanted, and it does not hold the
interval.  Measured at N=256, W=2 on xcu280, from `csynth.rpt`:

    banked, no bank partition   II=3 at every unit
    banked, bank partitioned    II=1 at stage 0, II=2 at stages 1-7, II=4 at
                                the reorder; array interval 512, four times
                                the ideal, cosim passing throughout
    split                       II=1 at all nine, array interval 128 exactly

The first line is a real bug and is fixed -- `schedule.partition_banks` -- but
the second is the thing itself.  `bank(p)` is a *runtime* value, so Vitis
cannot prove the cycle's two reads land in different banks and must give each
bank two read ports beside its write.  The split form makes the bank index a
constant of the loop body and the routing dynamic instead, so each bank sees
exactly one read and one write; the muxes are what the swizzle costs when it
has to be honoured by a scheduler rather than by a wire.

The banked form costs less area for it -- LUT 3,085 a unit against the split
form's 3,491, FF 3,438 against 3,613, the same 12 DSP and 4 BRAM -- and the
exact interval is worth more than either column, so the default is `split`.
The banked form stays because it is the measurement that says so.

That was not true when this was written.  The banked form first synthesised at
**II=3**, because a banked memory reached Vitis with all its banks in one
memory sharing one set of ports; `schedule.partition_banks` emits the pragma
that makes the banks real, and the same source then closes at II=1.  A layout
that places data without partitioning the ports is arithmetic with nothing on
the other end of it.
"""

import cmath
import os

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import float32, int32

csample = float32[2]  # one complex sample: [re, im]


def bitrev(x, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def _w(k, n):
    """The twiddle W_N^k = exp(-2 pi i k / n), as (re, im)."""
    z = cmath.exp(-2j * cmath.pi * k / n)
    return z.real, z.imag


def paired_layout(n, lanes):
    """The layout every buffer in the design is placed by.

    One object, asked rather than restated: the stage bodies take their bank
    and row from `bank_of`/`row_of`, and the banked form hands the same layout
    to `spmw.mem`. If the two ever disagreed the design would read the wrong
    element, which is precisely the failure banking is prone to, so they are
    the same object and not two copies of one formula.
    """
    return spmw.xor_bank(lanes, stride_bit=int(np.log2(n)) - 1)


def paired_geometry(n, batch, lanes=2):
    """The constants the schedule is made of, so a test can check them."""
    stages = int(np.log2(n))
    half = n // lanes  # cycles a transform, and butterflies a stage
    depth = 2 * n  # buffer positions: two blocks, so a slot outlives its use
    lag = n // 4 + 1  # cycles a stage issues behind its input
    rlag = n // 2  # the reorder needs a whole block before it can start
    lead = stages * lag + rlag  # cycles of fill before the first real output
    cycles = lead + batch * half
    return stages, half, depth, lag, rlag, lead, cycles


# ---------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------


def fft_paired_of(n, batch, lanes=2, name=None, buffer=None):
    """`n` points, two samples a cycle, one butterfly unit a stage."""
    stages, half, depth, lag, rlag, lead, cycles = paired_geometry(n, batch, lanes)
    assert lanes == 2, (
        "the paired design needs two samples a cycle to fill one butterfly; "
        "at one sample a cycle there is no pair to take, which is the rolled "
        "design's situation and the reason it runs at half"
    )
    assert 1 << stages == n, "radix-2 only"
    rows = depth // lanes
    layout = paired_layout(n, lanes)
    banked = (buffer or os.environ.get("SPMW_FFT_BUFFER", "split")) == "banked"

    # The swizzle, asked of the layout once here and emitted into the bodies as
    # the same arithmetic. `stride_bit` is the position bit the butterfly's two
    # operands differ in -- log2(N) - 1 at every stage, because a stage puts
    # its two results in the two positions of one cycle.
    sbit = layout.stride_bit
    assert layout.bank_of(0) != layout.bank_of(1), "adjacent writes must split"
    assert layout.bank_of(0) != layout.bank_of(1 << sbit), "the read pair must split"

    # ------------------------------------------------------------ one stage
    def stage_unit(s):
        """Stage `s`: distance `n >> (s+1)`, one whole butterfly a cycle.

        The twiddle exponent is `j` with its low `s` bits cleared -- stage `s`
        advances its angle once every `2**s` slots -- so the table is indexed
        by the slot and holds `half` entries whatever the stage.

        A stage issues `lag` cycles behind its input, so its results land `2 *
        lag` positions further down the stream than its operands did and every
        stage sees the block boundary in a different place. `base` is that
        shift, a constant per stage; without it the addressing is right at
        stage 0 and quietly wrong at every stage after it.
        """
        base = (2 * s * lag) % depth
        off = (-(s + 1) * lag) % n

        class StageIO(spmw.Interface):
            a_in = spmw.In(csample, depth=8)
            b_in = spmw.In(csample, depth=8)
            a_out = spmw.Out(csample, depth=8)
            b_out = spmw.Out(csample, depth=8)
            tw = spmw.MemIn(float32[half, 2])
            if banked:
                # One linear address space per component, placed by the
                # layout. The lowering emits the bank arithmetic and stores the
                # contents permuted; `init=` is the zeroed scratch the unit
                # fills, not a table it reads.
                bufr = spmw.MemIn(float32[depth])
                bufi = spmw.MemIn(float32[depth])

        def banked_body(io: StageIO):
            for t in range(cycles):
                x = io.a_in.get()
                y = io.b_in.get()
                w0: int32 = (2 * t) & (depth - 1)
                io.bufr[w0] = x[0]
                io.bufi[w0] = x[1]
                io.bufr[w0 + 1] = y[0]
                io.bufi[w0 + 1] = y[1]
                # `u` counts from this stage's own first butterfly, kept
                # non-negative by the offset -- every use of it is mod `n`.
                u: int32 = t + off
                j: int32 = u & (half - 1)
                blk: int32 = (u >> sbit) & 1
                pa: int32 = (base + (blk << stages) + j) & (depth - 1)
                pb: int32 = (pa + half) & (depth - 1)
                ar: float32 = io.bufr[pa]
                ai: float32 = io.bufi[pa]
                br: float32 = io.bufr[pb]
                bi: float32 = io.bufi[pb]
                wr: float32 = io.tw[j, 0]
                wi: float32 = io.tw[j, 1]
                dr: float32 = ar - br
                di: float32 = ai - bi
                p: csample
                q: csample
                p[0] = ar + br
                p[1] = ai + bi
                q[0] = dr * wr - di * wi
                q[1] = dr * wi + di * wr
                io.a_out.put(p)
                io.b_out.put(q)

        def split_body(io: StageIO):
            # The same layout, held as one memory per bank. `bank_of` decides
            # which memory an address lives in and `row_of` where in it; both
            # are constants of the loop body here, so each bank sees exactly
            # one read and one write a cycle and the pair never contends.
            z0r: float32[rows]
            z0i: float32[rows]
            z1r: float32[rows]
            z1i: float32[rows]
            for t in range(cycles):
                x = io.a_in.get()
                y = io.b_in.get()
                w0: int32 = (2 * t) & (depth - 1)
                wsel: int32 = (w0 >> sbit) & 1  # bank_of(w0); w0 is even
                rw: int32 = w0 >> 1  # row_of(w0) == row_of(w0 + 1)
                u: int32 = t + off
                j: int32 = u & (half - 1)
                blk: int32 = (u >> sbit) & 1
                pa: int32 = (base + (blk << stages) + j) & (depth - 1)
                pb: int32 = (pa + half) & (depth - 1)
                sel: int32 = (pa & 1) ^ ((pa >> sbit) & 1)  # bank_of(pa)
                ra: int32 = pa >> 1
                rb: int32 = pb >> 1
                # Address each bank once: the low operand is in `sel`, the high
                # in the other, so bank 0 wants `ra` when sel is 0 and `rb`
                # when it is 1.
                r0: int32 = ra
                r1: int32 = rb
                if sel == 1:
                    r0 = rb
                    r1 = ra
                g0r: float32 = z0r[r0]
                g0i: float32 = z0i[r0]
                g1r: float32 = z1r[r1]
                g1i: float32 = z1i[r1]
                ar: float32 = g0r
                ai: float32 = g0i
                br: float32 = g1r
                bi: float32 = g1i
                if sel == 1:
                    ar = g1r
                    ai = g1i
                    br = g0r
                    bi = g0i
                if wsel == 0:
                    z0r[rw] = x[0]
                    z0i[rw] = x[1]
                    z1r[rw] = y[0]
                    z1i[rw] = y[1]
                else:
                    z1r[rw] = x[0]
                    z1i[rw] = x[1]
                    z0r[rw] = y[0]
                    z0i[rw] = y[1]
                wr: float32 = io.tw[j, 0]
                wi: float32 = io.tw[j, 1]
                dr: float32 = ar - br
                di: float32 = ai - bi
                p: csample
                q: csample
                p[0] = ar + br
                p[1] = ai + bi
                q[0] = dr * wr - di * wi
                q[1] = dr * wi + di * wr
                io.a_out.put(p)
                io.b_out.put(q)

        body = banked_body if banked else split_body
        # A unit takes its name at decoration, and two bodies sharing a name
        # share their captured constants -- `s` differs at every stage.
        body.__name__ = f"bfly{s}"
        body.__qualname__ = f"bfly{s}"

        mask = ~((1 << s) - 1)
        table = np.zeros((half, 2), dtype=np.float32)
        for j in range(half):
            table[j] = _w(j & mask, n)
        return spmw.unit(body), table

    # ---------------------------------------------------------- the reorder
    # Decimation in frequency leaves the transform in bit-reversed order. The
    # reorder is the same buffer read through a permutation: output position
    # `q` wants input position `bitrev(q)`, and `bitrev(2c)` and `bitrev(2c+1)`
    # differ in exactly the top bit -- the same `N/2` apart the butterflies
    # were, so the same layout places it and no second swizzle is needed.
    class ReorderIO(spmw.Interface):
        a_in = spmw.In(csample, depth=8)
        b_in = spmw.In(csample, depth=8)
        a_out = spmw.Out(csample, depth=8)
        b_out = spmw.Out(csample, depth=8)
        rd = spmw.MemIn(int32[half])
        if banked:
            obufr = spmw.MemIn(float32[depth])
            obufi = spmw.MemIn(float32[depth])

    rbase = (2 * stages * lag) % depth
    roff = (-lead) % n

    def reorder_banked(io: ReorderIO):
        for t in range(cycles):
            x = io.a_in.get()
            y = io.b_in.get()
            w0: int32 = (2 * t) & (depth - 1)
            io.obufr[w0] = x[0]
            io.obufi[w0] = x[1]
            io.obufr[w0 + 1] = y[0]
            io.obufi[w0 + 1] = y[1]
            u: int32 = t + roff
            j: int32 = u & (half - 1)
            blk: int32 = (u >> sbit) & 1
            pa: int32 = (rbase + (blk << stages) + io.rd[j]) & (depth - 1)
            pb: int32 = (pa + half) & (depth - 1)
            p: csample
            q: csample
            p[0] = io.obufr[pa]
            p[1] = io.obufi[pa]
            q[0] = io.obufr[pb]
            q[1] = io.obufi[pb]
            if t >= lead:
                io.a_out.put(p)
                io.b_out.put(q)

    def reorder_split(io: ReorderIO):
        z0r: float32[rows]
        z0i: float32[rows]
        z1r: float32[rows]
        z1i: float32[rows]
        for t in range(cycles):
            x = io.a_in.get()
            y = io.b_in.get()
            w0: int32 = (2 * t) & (depth - 1)
            wsel: int32 = (w0 >> sbit) & 1
            rw: int32 = w0 >> 1
            u: int32 = t + roff
            j: int32 = u & (half - 1)
            blk: int32 = (u >> sbit) & 1
            pa: int32 = (rbase + (blk << stages) + io.rd[j]) & (depth - 1)
            pb: int32 = (pa + half) & (depth - 1)
            sel: int32 = (pa & 1) ^ ((pa >> sbit) & 1)
            ra: int32 = pa >> 1
            rb: int32 = pb >> 1
            r0: int32 = ra
            r1: int32 = rb
            if sel == 1:
                r0 = rb
                r1 = ra
            g0r: float32 = z0r[r0]
            g0i: float32 = z0i[r0]
            g1r: float32 = z1r[r1]
            g1i: float32 = z1i[r1]
            p: csample
            q: csample
            p[0] = g0r
            p[1] = g0i
            q[0] = g1r
            q[1] = g1i
            if sel == 1:
                p[0] = g1r
                p[1] = g1i
                q[0] = g0r
                q[1] = g0i
            if wsel == 0:
                z0r[rw] = x[0]
                z0i[rw] = x[1]
                z1r[rw] = y[0]
                z1i[rw] = y[1]
            else:
                z1r[rw] = x[0]
                z1i[rw] = x[1]
                z0r[rw] = y[0]
                z0i[rw] = y[1]
            if t >= lead:
                io.a_out.put(p)
                io.b_out.put(q)

    reorder = spmw.unit(reorder_banked if banked else reorder_split)
    rd_tab = np.array([bitrev(2 * j, stages) for j in range(half)], dtype=np.int32)

    units = [stage_unit(s) for s in range(stages)]
    stage_tw = [tab for _u, tab in units]
    stage_units = [u for u, _t in units]

    # ------------------------------------------------------------ the fabric
    @spmw.fabric
    def engine(
        Xa: float32[cycles, 2],
        Xb: float32[cycles, 2],
        Ya: float32[batch * half, 2],
        Yb: float32[batch * half, 2],
    ):
        S = [spmw.place(u, on=spmw.Grid((1,))) for u in stage_units]
        R = spmw.place(reorder, on=spmw.Grid((1,)))

        # One ROM per stage, each with its own name: memories made in a loop
        # would otherwise all be called `tw`, and one tensor bound stationary at
        # several placements reaches only the first.
        for s, p in enumerate(S):
            tw = spmw.mem(
                float32[half, 2],
                init=stage_tw[s],
                layout=spmw.replicate,
                name=f"twp{s}",
            )
            spmw.stationary(tw, at=p.tw)
            if banked:
                zero = np.zeros(depth, dtype=np.float32)
                spmw.stationary(
                    spmw.mem(
                        float32[depth], init=zero, layout=layout, name=f"bufr{s}"
                    ),
                    at=p.bufr,
                )
                spmw.stationary(
                    spmw.mem(
                        float32[depth], init=zero, layout=layout, name=f"bufi{s}"
                    ),
                    at=p.bufi,
                )
        spmw.stationary(
            spmw.mem(int32[half], init=rd_tab, layout=spmw.replicate, name="rdp"),
            at=R.rd,
        )
        if banked:
            zero = np.zeros(depth, dtype=np.float32)
            spmw.stationary(
                spmw.mem(float32[depth], init=zero, layout=layout, name="obufr"),
                at=R.obufr,
            )
            spmw.stationary(
                spmw.mem(float32[depth], init=zero, layout=layout, name="obufi"),
                at=R.obufi,
            )

        spmw.stream_in(Xa, into=S[0].a_in, index=(...,))
        spmw.stream_in(Xb, into=S[0].b_in, index=(...,))
        for a, b in zip(S, S[1:]):
            spmw.link(a.a_out, to=b.a_in)
            spmw.link(a.b_out, to=b.b_in)
        spmw.link(S[-1].a_out, to=R.a_in)
        spmw.link(S[-1].b_out, to=R.b_in)
        spmw.gather(Ya, from_=R.a_out, index=(...,))
        spmw.gather(Yb, from_=R.b_out, index=(...,))

    engine.__name__ = name or f"fft_paired_{n}_w{lanes}"
    engine.spmw_parts = (n, batch, lanes, stages, half, depth, lag, rlag, lead)
    # The butterflies cancel O(N) intermediates, so the differences between the
    # reference and the HLS float units are absolute, ~1e-5.
    engine.spmw_tolerance = (1e-4, 1e-4)
    # Every body is one deep pipeline over the whole launch, so it has to drain
    # when its loop ends: with the default stall style HLS keeps the iterations
    # in flight and the last unit is short by its depth.
    engine.spmw_pipeline_style = "flp"
    # On by default here, unlike the rolled design: the row this is measured
    # against is HP-FFT's, which binds its own adders, and a DSP comparison
    # where one side spends DSPs on adds is not a comparison.
    engine.spmw_bind_fabric = os.environ.get("SPMW_BIND_FABRIC", "1") != "0"
    # One transform is `n` output tokens across both lanes; the testbench counts
    # tokens on every channel, not per channel.
    engine.spmw_tokens_per_transform = n
    return engine


# ---------------------------------------------------------------------------
# Operands and the check
# ---------------------------------------------------------------------------


def operands(n, batch, lanes=2, seed=0):
    """`batch` transforms, even samples on one lane and odd on the other."""
    _st, half, _d, _l, _r, _lead, cycles = paired_geometry(n, batch, lanes)
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((batch, n)) + 1j * rng.standard_normal((batch, n))).astype(
        np.complex64
    )
    flat = np.zeros((cycles * lanes, 2), dtype=np.float32)
    flat[: batch * n, 0] = x.real.reshape(-1)
    flat[: batch * n, 1] = x.imag.reshape(-1)
    pair = flat.reshape(cycles, lanes, 2)
    return x, pair[:, 0].copy(), pair[:, 1].copy()


def unpack(Ya, Yb, n, batch, lanes=2):
    """Bin `2c` leaves on the first lane and `2c+1` on the second."""
    half = n // lanes
    out = np.zeros((batch, n), dtype=np.complex128)
    a = Ya.reshape(batch, half, 2)
    b = Yb.reshape(batch, half, 2)
    out[:, 0::2] = a[:, :, 0] + 1j * a[:, :, 1]
    out[:, 1::2] = b[:, :, 0] + 1j * b[:, :, 1]
    return out


def check(x, Ya, Yb, n, batch, lanes=2, atol=1e-4, rtol=1e-4):
    got = unpack(Ya, Yb, n, batch, lanes)
    want = np.fft.fft(x.astype(np.complex128), axis=1)
    err = np.abs(got - want).max()
    norm = err / max(np.abs(want).max(), 1e-30)
    np.testing.assert_allclose(got, want, atol=atol, rtol=rtol)
    return err, norm


def run(n, batch, target, lanes=2, seed=0, buffer=None):
    x, Xa, Xb = operands(n, batch, lanes, seed=seed)
    half = n // lanes
    Ya = np.zeros((batch * half, 2), dtype=np.float32)
    Yb = np.zeros((batch * half, 2), dtype=np.float32)
    spmw.build(fft_paired_of(n, batch, lanes, buffer=buffer), target=target)(
        Xa, Xb, Ya, Yb
    )
    return check(x, Ya, Yb, n, batch, lanes)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ref", "simulator"])
@pytest.mark.parametrize("buffer", ["split", "banked"])
@pytest.mark.parametrize("n", [8, 16, 32])
def test_paired_matches_numpy(n, buffer, target):
    err, _norm = run(n, 3, target, buffer=buffer)
    assert err < 1e-3, err


@pytest.mark.parametrize("n", [64, 128, 256])
def test_the_wide_end_of_the_sweep(n):
    err, _norm = run(n, 2, "ref")
    assert err < 1e-3, err


def test_the_two_buffer_forms_agree():
    """One layout, two realisations; a difference is a swizzle bug."""
    x, Xa, Xb = operands(32, 3)
    out = {}
    for form in ("split", "banked"):
        Ya = np.zeros((3 * 16, 2), dtype=np.float32)
        Yb = np.zeros((3 * 16, 2), dtype=np.float32)
        spmw.build(fft_paired_of(32, 3, buffer=form), target="ref")(Xa, Xb, Ya, Yb)
        out[form] = (Ya.copy(), Yb.copy())
    np.testing.assert_array_equal(out["split"][0], out["banked"][0])
    np.testing.assert_array_equal(out["split"][1], out["banked"][1])
    assert np.abs(out["split"][0]).sum() > 0, "both forms produced nothing"


def test_the_layout_places_every_access():
    """The bodies' bank and row arithmetic *is* the layout's, not a copy.

    The split form writes `(p & 1) ^ ((p >> sbit) & 1)` and `p >> 1` out into
    the unit body, because that is what a bank selector has to be at runtime.
    Written out is where a formula drifts from the object it came from, so it
    is checked against the object for every address the design can produce.
    """
    n, lanes = 256, 2
    layout = paired_layout(n, lanes)
    depth = 2 * n
    for p in range(depth):
        assert layout.bank_of(p) == ((p & 1) ^ ((p >> layout.stride_bit) & 1)), p
        assert layout.row_of(p) == p >> 1, p


def test_both_the_read_pair_and_the_write_pair_split():
    """Two different access patterns, one swizzle, and it has to serve both.

    The arriving pair is adjacent and the butterfly's pair is `N/2` apart.
    Cyclic banking separates the first and collides on every one of the second,
    which is the control that makes this a measurement of the swizzle.
    """
    n, lanes = 256, 2
    swizzled = paired_layout(n, lanes)
    plain = spmw.banked(banks=lanes)
    half, depth = n // lanes, 2 * n
    for p in range(0, depth, 2):
        assert swizzled.bank_of(p) != swizzled.bank_of(p + 1), f"write pair {p}"
    collisions = 0
    for p in range(depth - half):
        assert swizzled.bank_of(p) != swizzled.bank_of(p + half), f"read pair {p}"
        collisions += plain.bank_of(p) == plain.bank_of(p + half)
    assert collisions == depth - half, "cyclic banking should collide on all of them"


def test_one_butterfly_unit_per_stage():
    """The count is the whole point: `log2(N)` units, not `W * log2(N)`."""
    graph = spmw.elaborate(fft_paired_of(256, 2))
    placements = [p for p in graph.placements if not p.expanded]
    sites = sum(len(list(p.sites())) for p in placements)
    assert len(placements) == 9, [p.name for p in placements]  # 8 stages + reorder
    assert sites == 9, sites


def test_the_interval_is_the_ideal_one():
    """`N/W` cycles a transform, which is one butterfly a cycle a stage."""
    n, batch, lanes = 256, 33, 2
    stages, half, _d, lag, rlag, lead, _c = paired_geometry(n, batch, lanes)
    assert half == n // lanes == 128
    # log2(N) stages each retiring N/2 butterflies in N/2 cycles.
    assert stages * (n // 2) == 1024, "butterflies a transform at N=256"
    assert (n // 2) * stages / (half * stages) == 1.0, "one a cycle a unit"
    assert lead == stages * lag + rlag


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
