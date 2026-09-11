# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A rolled radix-2 FFT: the unroll factor is where a butterfly's partner lives.

`test_spmw_fft_sdf.py` is one design point -- a single-path delay-feedback
pipeline, one complex sample a cycle. HP-FFT ships six (UF1..UF32, a datapath of
``2*UF`` complex samples a beat), so comparing them means comparing SPMW's one
configuration against HP-FFT's narrowest. This is the family that closes that
gap: `W` lanes, `W` complex samples a cycle, and `W` is a parameter.

**The lane law.** The `N` points of a block are spread over `W` lanes by
``lane(i) = i & (W - 1)``, ``row(i) = i >> log2(W)`` -- `spmw.banked(banks=W)`'s
own ``bank_of``/``row_of``, and the same law at every stage. Stage `s` of the
decimation-in-frequency recursion pairs `i` with ``i ^ (1 << d)``, ``d = S-1-s``,
so the law decides where that partner is:

- ``d >= log2(W)``: the partner is **the same lane at another time**,
  ``1 << (d - w)`` rows away. There is no wire to name; the partner is a delay
  line. This is `test_spmw_fft_sdf.py`'s stage, once per lane.
- ``d < log2(W)``: the partner is **another lane at the same time**, lane
  ``l ^ (1 << d)``. That is a wire, and it is named as one: the stage's
  topology is ``link=lambda l: {p_out: to((l ^ stride,), p_in)}``.

So **the unroll factor is the space/time split of the butterfly partners**:
``log2(W)`` of the ``log2(N)`` stages have their partners in space, the rest in
time. At ``W = 1`` every partner is a delay line and this design *is* the folded
SDF pipeline; at ``W = N/2`` every partner is a wire.

Radix-2, complex FP32, natural-order input and output, unnormalised forward
transform; `batch` transforms a launch, back to back.

**Why the twiddles are laid out the way they are.** A resident ROM's contents
are the same at every site of its placement -- ``spmw.stationary(brick, at=...,
index=...)`` accepts a per-site index map and no path slices by it, so a ROM
that differs per lane is not expressible. The delay stage therefore holds its
whole stage's table and indexes it by the lane, which costs `W` times the ROM
it reads; the cross stage's table is genuinely lane-independent (its stride is
below the lane count, so the twiddle is a function of ``l & (stride-1)`` alone)
and costs nothing. `test_stationary_index_on_a_brick_is_ignored` pins the gap
that forces the first of those.
"""

import cmath

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
    """The twiddle W_N^k = exp(-2 pi i k / N), as (re, im)."""
    z = cmath.exp(-2j * cmath.pi * k / n)
    return z.real, z.imag


def lane_law(n, w_bits):
    """Where each stage's butterfly partner lives, read out of the layout.

    The lane assignment is a `spmw.Layout`, not arithmetic inlined into a body:
    ``bank_of`` is the lane and ``row_of`` is the row. Deriving the split from
    the layout is what keeps the topologies' `link` rules and the delay-line
    spans describing one thing.
    """
    layout = spmw.banked(banks=1 << w_bits)
    stages = int(np.log2(n))
    split = []
    for s in range(stages):
        d = stages - 1 - s
        # The pair (i, i ^ (1 << d)) for an i with bit d clear; the layout says
        # whether the partner is another lane or another row of the same one.
        crosses = layout.bank_of(0) != layout.bank_of(1 << d)
        split.append(("cross" if crosses else "delay", d))
    return layout, split


def fft_rolled_of(n, batch, lanes, name=None):
    """`n` points over `lanes` lanes, `batch` transforms a launch."""
    stages = int(np.log2(n))
    assert 1 << stages == n, "radix-2 only"
    assert lanes & (lanes - 1) == 0, "the lane count is a power of two"
    w_bits = int(np.log2(lanes))
    assert w_bits < stages, "at least one stage must keep its partner in time"

    rows = n // lanes  # rows a lane holds, per transform
    total_rows = (batch + 1) * rows  # one extra block flushes the delay lines
    n_delay = stages - w_bits  # stages whose partner is a delay line
    n_cross = w_bits  # stages whose partner is a wire

    # ------------------------------------------------------------ the delay
    # One placement per delay stage, on a grid of lanes. The span is a literal
    # in the body, so the delay line is sized exactly: a specialised grid axis
    # cannot do this, because on the dataflow path the coordinate stays a
    # `get_pid()` and `float32[span]` is then not a constant.
    #
    # Each stage is built in its own function call, not in a loop body: a body
    # defined in a loop closes over the loop's *variable*, so every stage would
    # see the last stage's `span`. A call gives each one its own cell.
    def delay_stage(s):
        span = 1 << (stages - 1 - w_bits - s)  # rows in this lane's half-block
        blocks = (batch + 1) << s

        class StageIO(spmw.Interface):
            x_in = spmw.In(csample)
            x_out = spmw.Out(csample)
            # This stage's whole twiddle table, indexed by (position, lane).
            tw = spmw.MemIn(float32[span, lanes, 2])

        def stage(io: StageIO, site: spmw.Site):
            """Stage `s` of one lane: the partner is `span` rows back.

            Computed on the way *out*, as in the folded pipeline: a block's
            twiddled differences leave during the next block's first half from
            the stored inputs, its sums during its own second half as the
            partner arrives. Nothing computed is stored, so the float pipeline
            is feed-forward and the nest closes at one token a cycle; a delay
            line that stored the twiddled difference would carry a
            read-modify-write recurrence the short spans cannot close.
            """
            (ell,) = site.rank
            ar: float32[span]
            ai: float32[span]
            br: float32[span]
            bi: float32[span]
            for _b in range(blocks):
                for h in range(2):
                    for c in range(span):
                        x = io.x_in.get()
                        y: csample
                        if h == 0:
                            dr: float32 = ar[c] - br[c]
                            di: float32 = ai[c] - bi[c]
                            wr: float32 = io.tw[c, ell, 0]
                            wi: float32 = io.tw[c, ell, 1]
                            y[0] = dr * wr - di * wi
                            y[1] = dr * wi + di * wr
                            ar[c] = x[0]
                            ai[c] = x[1]
                        else:
                            y[0] = ar[c] + x[0]
                            y[1] = ai[c] + x[1]
                            br[c] = x[0]
                            bi[c] = x[1]
                        io.x_out.put(y)

        # A unit takes its name at decoration, and two bodies sharing a name
        # share their captured constants: rename first, then decorate.
        stage.__name__ = f"delay{s}"
        stage.__qualname__ = f"delay{s}"

        tab = np.zeros((span, lanes, 2), dtype=np.float32)
        for c in range(span):
            for ell in range(lanes):
                # the pair's position within the stage's span is c*W + l
                tab[c, ell] = _w(((c * lanes + ell) << s), n)
        return spmw.unit(stage), tab

    delay_units, delay_tw = [], []
    for _s in range(n_delay):
        _u, _t = delay_stage(_s)
        delay_units.append(_u)
        delay_tw.append(_t)

    # ------------------------------------------------------------ the cross
    # One placement per cross stage, on a topology whose `link` rule names each
    # butterfly's partner. The twiddle here really is lane-independent -- the
    # stride is below the lane count, so a lane's twiddle is a function of
    # `l & (stride - 1)` alone -- so the table is exact at every site.
    def cross_stage(t):
        s = n_delay + t
        d = w_bits - 1 - t  # this stage's separating bit
        stride = 1 << d

        class CrossIO(spmw.Interface):
            x_in = spmw.In(csample)
            x_out = spmw.Out(csample)
            p_in = spmw.In(csample, depth=2)
            p_out = spmw.Out(csample, depth=2)
            tw = spmw.MemIn(float32[stride, 2])

        topo = spmw.Topology(
            CrossIO,
            grid=(lanes,),
            # The butterfly's partner, named. Nothing here is an index into a
            # shared buffer: the two operands are two units, and the XOR is the
            # wire between them.
            link=lambda ell: {CrossIO.p_out: spmw.to((ell ^ stride,), CrossIO.p_in)},
            name=f"cross{t}",
        )

        def cross(io: CrossIO, site: spmw.Site):
            """One butterfly a cycle, its partner arriving from the named lane.

            Which half of the pair a lane holds is a bit of its own
            coordinate, so it is arithmetic rather than a per-site constant --
            and the twiddle is the same table at every lane of the stage.
            """
            (ell,) = site.rank
            # Annotated, not inferred: an unannotated local holding a
            # comparison is given the type of the arithmetic that feeds it and
            # then stored as a predicate, which the dataflow lowering rejects
            # as `affine.store ... must have the same type as memref element`.
            half: int32 = (ell >> d) & 1
            c: int32 = ell & (stride - 1)
            for _r in range(total_rows):
                x = io.x_in.get()
                io.p_out.put(x)
                y = io.p_in.get()
                o: csample
                if half == 0:  # I hold the upper operand: the sum is mine
                    o[0] = x[0] + y[0]
                    o[1] = x[1] + y[1]
                else:  # I hold the lower: the twiddled difference is mine
                    dr: float32 = y[0] - x[0]
                    di: float32 = y[1] - x[1]
                    wr: float32 = io.tw[c, 0]
                    wi: float32 = io.tw[c, 1]
                    o[0] = dr * wr - di * wi
                    o[1] = dr * wi + di * wr
                io.x_out.put(o)

        cross.__name__ = f"cross{t}"
        cross.__qualname__ = f"cross{t}"

        tab = np.zeros((stride, 2), dtype=np.float32)
        for c in range(stride):
            tab[c] = _w(c << s, n)
        return spmw.unit(cross), topo, tab

    cross_units, cross_topos, cross_tw = [], [], []
    for _t in range(n_cross):
        _u, _topo, _tab = cross_stage(_t)
        cross_units.append(_u)
        cross_topos.append(_topo)
        cross_tw.append(_tab)

    # ---------------------------------------------------------- the reorder
    # Decimation in frequency leaves each lane's own `rows` points bit-reversed
    # among themselves: position r*W + l carries bin bitrev_w(l)*R +
    # bitrev_{S-w}(r), so lane `l` owns a contiguous output block and the
    # reordering never crosses a lane.
    class ReorderIO(spmw.Interface):
        x_in = spmw.In(csample)
        y_out = spmw.Out(csample)
        perm = spmw.MemIn(int32[rows])

    @spmw.unit
    def reorder(io: ReorderIO):
        # The lane's first `rows - 1` tokens are the delay lines' initial
        # contents -- their spans sum to 2^(S-w) - 1 -- and then `batch`
        # blocks. One loop drives both halves of the double buffer, so a block
        # costs `rows` cycles and not two.
        bufr: float32[2, rows]
        bufi: float32[2, rows]
        for _t in range(rows - 1):
            _skip = io.x_in.get()
        for b in range(batch):
            side: int32 = b & 1
            other: int32 = 1 - side
            for i in range(rows):
                x = io.x_in.get()
                p: int32 = io.perm[i]
                bufr[side, p] = x[0]
                bufi[side, p] = x[1]
                if b > 0:
                    y: csample
                    y[0] = bufr[other, i]
                    y[1] = bufi[other, i]
                    io.y_out.put(y)
        last: int32 = (batch - 1) & 1
        for i in range(rows):
            y2: csample
            y2[0] = bufr[last, i]
            y2[1] = bufi[last, i]
            io.y_out.put(y2)

    perm_tab = np.array(
        [bitrev(r, stages - w_bits) for r in range(rows)], dtype=np.int32
    )

    # ----------------------------------------------------------- the fabric
    @spmw.fabric
    def engine(X: float32[lanes, total_rows, 2], Y: float32[lanes, batch * rows, 2]):
        D = [spmw.place(u, on=spmw.Grid((lanes,))) for u in delay_units]
        C = [spmw.place(u, on=topo) for u, topo in zip(cross_units, cross_topos)]
        R = spmw.place(reorder, on=spmw.Grid((lanes,)))

        # One ROM per stage, each with its own name: memories made in a loop
        # would otherwise all be called `tw`, and one tensor bound stationary at
        # several placements reaches only the first.
        for s, p in enumerate(D):
            span = 1 << (stages - 1 - w_bits - s)
            tw = spmw.mem(
                float32[span, lanes, 2],
                init=delay_tw[s],
                layout=spmw.replicate,
                name=f"twd{s}",
            )
            spmw.stationary(tw, at=p.tw)
        for t, p in enumerate(C):
            stride = 1 << (w_bits - 1 - t)
            tw = spmw.mem(
                float32[stride, 2],
                init=cross_tw[t],
                layout=spmw.replicate,
                name=f"twc{t}",
            )
            spmw.stationary(tw, at=p.tw)
        perm = spmw.mem(int32[rows], init=perm_tab, layout=spmw.replicate)
        spmw.stationary(perm, at=R.perm)

        # Lane `l` carries the samples whose index is congruent to `l`: the
        # lane law, at the boundary.
        spmw.stream_in(X, into=D[0].x_in, index=(D[0].rows, ...))
        chain = D + C
        for a, b in zip(chain, chain[1:]):
            spmw.link(a.x_out, to=b.x_in)
        spmw.link(chain[-1].x_out, to=R.x_in)
        spmw.gather(Y, from_=R.y_out, index=(R.rows, ...))

    engine.__name__ = name or f"fft_rolled_{n}_w{lanes}"
    engine.spmw_parts = (n, batch, lanes, stages, n_delay, n_cross, rows)
    # The butterflies cancel O(N) intermediates, so the differences between the
    # reference and the HLS float units are absolute, ~1e-5.
    engine.spmw_tolerance = (1e-4, 1e-4)
    # Every body is one deep pipeline over the whole launch, so it has to drain
    # when its loop ends: with the default stall style HLS keeps the iterations
    # in flight and the last unit is short by its depth.
    engine.spmw_pipeline_style = "flp"
    # One transform is `rows` tokens out of each lane.
    engine.spmw_tokens_per_transform = rows
    return engine


# ---------------------------------------------------------------------------
# Operands and the check
# ---------------------------------------------------------------------------


def operands(n, batch, lanes, seed=0):
    """`batch` transforms, laid out over the lanes by the lane law."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((batch, n)) + 1j * rng.standard_normal((batch, n))).astype(
        np.complex64
    )
    rows = n // lanes
    flat = np.zeros(((batch + 1) * n, 2), dtype=np.float32)
    flat[: batch * n, 0] = x.real.reshape(-1)
    flat[: batch * n, 1] = x.imag.reshape(-1)
    # sample t*W + l of the stream is lane l, row t
    X = flat.reshape((batch + 1) * rows, lanes, 2).transpose(1, 0, 2).copy()
    return x, X


def unpack(Y, n, batch, lanes):
    """Undo the lane split on the output: lane `l` owns a contiguous block."""
    rows = n // lanes
    w_bits = int(np.log2(lanes))
    out = np.zeros((batch, n), dtype=np.complex128)
    for ell in range(lanes):
        base = bitrev(ell, w_bits) * rows
        block = Y[ell].reshape(batch, rows, 2)
        out[:, base : base + rows] = block[:, :, 0] + 1j * block[:, :, 1]
    return out


def check(x, Y, n, batch, lanes, atol=1e-4, rtol=1e-4):
    got = unpack(Y, n, batch, lanes)
    want = np.fft.fft(x.astype(np.complex128), axis=1)
    err = np.abs(got - want).max()
    norm = err / max(np.abs(want).max(), 1e-30)
    np.testing.assert_allclose(got, want, atol=atol, rtol=rtol)
    return err, norm


def run(n, batch, lanes, target, seed=0):
    x, X = operands(n, batch, lanes, seed=seed)
    Y = np.zeros((lanes, batch * (n // lanes), 2), dtype=np.float32)
    spmw.build(fft_rolled_of(n, batch, lanes), target=target)(X, Y)
    return check(x, Y, n, batch, lanes)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lanes", [1, 2, 4])
@pytest.mark.parametrize("n", [8, 16])
@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_rolled_matches_numpy(n, lanes, target):
    _err, norm = run(n, 3, lanes, target, seed=n * 16 + lanes)
    assert norm < 1e-5


def test_the_unroll_factor_is_where_the_partner_lives():
    """The claim the whole family rests on, read out of the layout itself.

    ``log2(W)`` stages have their butterfly partner in space and the rest in
    time. If that ever stops holding, the `link` rules and the delay-line spans
    have drifted apart and one of them is wrong.
    """
    for n in (16, 64, 256, 1024):
        stages = int(np.log2(n))
        for w_bits in range(0, stages):
            _layout, split = lane_law(n, w_bits)
            kinds = [k for k, _d in split]
            assert kinds.count("cross") == w_bits, (n, w_bits, kinds)
            assert kinds.count("delay") == stages - w_bits, (n, w_bits, kinds)


def test_at_one_lane_it_is_the_folded_pipeline():
    """W=1 is the design E2 already measured: every partner is a delay line."""
    fab = fft_rolled_of(128, 2, 1)
    _n, _b, lanes, stages, n_delay, n_cross, rows = fab.spmw_parts
    assert (lanes, n_delay, n_cross, rows) == (1, stages, 0, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
