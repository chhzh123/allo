# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A folded FFT: the radix-2 single-path delay-feedback pipeline in SPMW.

The spatial FFT (`test_spmw_fft.py`) is one butterfly per (stage, pair): N/2
butterflies a stage, which at 256 points is more DSPs than the part has. This
is the classic fold of it. One unit per stage, log2(N) units in a chain, each
holding a delay line of N / 2^(s+1) complex samples: for the first half of a
block it stores the incoming sample and emits what the delay line held (the
previous block's difference terms); for the second half it emits the sum of
the stored and incoming samples and stores their difference, twiddled. One
complex sample enters and one leaves every cycle; a transform costs N cycles
of interval, and the pipeline's latency is the sum of the delay lines, N - 1
samples. Decimation in frequency puts the output in bit-reversed order; a
last unit with a double buffer writes each block back in natural order.

Radix-2, complex FP32, natural-order input and output, unnormalised forward
transform: `BATCH` transforms per launch, back to back.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import float32, int32

csample = float32[2]


def twiddles(n):
    k = np.arange(n // 2)
    return np.stack(
        [np.cos(-2 * np.pi * k / n), np.sin(-2 * np.pi * k / n)], axis=1
    ).astype(np.float32)


def bitrev(x, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def fft_sdf_of(n, batch, name=None):
    """The pipeline for `n` points, `batch` transforms a launch."""
    stages = int(np.log2(n))
    assert 1 << stages == n
    half = n // 2
    total = (batch + 1) * n  # one extra block flushes the delay lines
    rev = np.array([bitrev(i, stages) for i in range(n)], dtype=np.int32)

    class StageIO(spmw.Interface):
        x_in = spmw.In(csample)
        x_out = spmw.Out(csample)
        tw = spmw.MemIn(float32[half, 2])

    class ReorderIO(spmw.Interface):
        x_in = spmw.In(csample)
        y_out = spmw.Out(csample)
        perm = spmw.MemIn(int32[n])

    def stage_unit(s):
        """Stage `s` as its own unit, every bound a literal.

        The stage keeps both halves of the current block -- a (first half)
        and b (second half) -- and computes on the way *out*: a block's
        (a - b) w leaves during the next block's first half from the stored
        inputs, a + b during the second half as b arrives. Nothing computed is
        stored, so the float pipeline is feed-forward and the nest pipelines
        at one token a cycle; a delay line that stores the twiddled
        difference has a read-modify-write recurrence that the last stages
        (span 1, 2, 4) cannot close in one cycle. The cost is a second buffer
        (N samples a stage rather than N/2, the same as HP-FFT's double
        buffers). Bounds are Python constants, so Allo labels the loops and
        the schedule pipelines and flattens them.
        """
        span = n >> (s + 1)  # this stage's half-block
        blocks = (batch + 1) << s  # blocks of 2 * span in (batch + 1) * n tokens

        def stage(io: StageIO):
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
                            # first half: out (a - b) w of the block before; keep a
                            dr: float32 = ar[c] - br[c]
                            di: float32 = ai[c] - bi[c]
                            k: int32 = c << s
                            wr: float32 = io.tw[k, 0]
                            wi: float32 = io.tw[k, 1]
                            y[0] = dr * wr - di * wi
                            y[1] = dr * wi + di * wr
                            ar[c] = x[0]
                            ai[c] = x[1]
                        else:
                            # second half: out a + b; keep b for the next first half
                            y[0] = ar[c] + x[0]
                            y[1] = ai[c] + x[1]
                            br[c] = x[0]
                            bi[c] = x[1]
                        io.x_out.put(y)

        # The unit takes its name from the function at decoration, and the
        # lowering names kernels after units: rename first, then decorate.
        stage.__name__ = f"stage{s}"
        stage.__qualname__ = f"stage{s}"
        return spmw.unit(stage)

    units = [stage_unit(s) for s in range(stages)]

    @spmw.unit
    def reorder(io: ReorderIO):
        # The first n - 1 tokens are the delay lines' initial contents; then
        # `batch` blocks. One loop does both halves of the double buffer: block
        # b lands bit-reversed on side b & 1 while block b - 1 streams out of
        # the other side in natural order, so a block costs n cycles, not 2n.
        bufr: float32[2, n]
        bufi: float32[2, n]
        for _t in range(n - 1):
            _skip = io.x_in.get()
        for b in range(batch):
            side: int32 = b & 1
            other: int32 = 1 - side
            for i in range(n):
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
        for i in range(n):
            y2: csample
            y2[0] = bufr[last, i]
            y2[1] = bufi[last, i]
            io.y_out.put(y2)

    @spmw.fabric
    def engine(X: float32[total, 2], Y: float32[batch * n, 2]):
        P = [spmw.place(u, on=spmw.Grid((1,))) for u in units]
        R = spmw.place(reorder, on=spmw.Grid((1,)))
        # One memory per stage, each with its own name: memories made in a
        # loop would otherwise all be called `tw`, and one tensor bound
        # stationary at several placements reaches only the first.
        for s, p in enumerate(P):
            tw = spmw.mem(
                float32[half, 2],
                init=twiddles(n),
                layout=spmw.replicate,
                name=f"tw{s}",
            )
            spmw.stationary(tw, at=p.tw)
        perm = spmw.mem(int32[n], init=rev, layout=spmw.replicate)
        spmw.stationary(perm, at=R.perm)
        spmw.stream_in(X, into=P[0].x_in, index=(...,))
        for s in range(stages - 1):
            spmw.link(P[s].x_out, to=P[s + 1].x_in)
        spmw.link(P[stages - 1].x_out, to=R.x_in)
        spmw.gather(Y, from_=R.y_out, index=(...,))

    engine.__name__ = name or f"fft_sdf_{n}"
    engine.spmw_parts = (units, reorder, n, batch, stages)
    # The RTL check's floating-point tolerance (relative, absolute): the
    # butterflies cancel O(N) intermediates, so the rounding differences
    # between the reference and the HLS float units are absolute, ~1e-5.
    engine.spmw_tolerance = (1e-4, 1e-4)
    # Each stage's body is one deep pipeline over the whole launch, so it has
    # to drain when its loop ends: with the default stall style HLS keeps the
    # iterations still in flight and the last stage is short by its depth
    # (17 of 4,352 tokens at N=128).
    engine.spmw_pipeline_style = "flp"
    # One transform is `n` output samples, so the cosim can report each
    # transform's completion cycle and not only the whole launch's. HP-FFT's
    # own harness reports a single transform, and comparing its figure with a
    # 33-transform launch is a factor of thirty error.
    engine.spmw_tokens_per_transform = n
    return engine


def operands(n, batch, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((batch, n)) + 1j * rng.standard_normal((batch, n))).astype(
        np.complex64
    )
    X = np.zeros(((batch + 1) * n, 2), dtype=np.float32)
    X[: batch * n, 0] = x.real.reshape(-1)
    X[: batch * n, 1] = x.imag.reshape(-1)
    return x, X


def check(x, Y, n, batch, atol=1e-4, rtol=1e-4):
    got = (Y[:, 0] + 1j * Y[:, 1]).reshape(batch, n)
    want = np.fft.fft(x.astype(np.complex128), axis=1)
    err = np.abs(got - want).max()
    norm = err / max(np.abs(want).max(), 1e-30)
    np.testing.assert_allclose(got, want, atol=atol, rtol=rtol)
    return err, norm


@pytest.mark.parametrize("n", [8, 16])
@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_sdf_matches_numpy(n, target):
    batch = 3
    x, X = operands(n, batch, seed=n)
    Y = np.zeros((batch * n, 2), dtype=np.float32)
    spmw.build(fft_sdf_of(n, batch), target=target)(X, Y)
    err, norm = check(x, Y, n, batch)
    assert norm < 1e-5


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_sdf_impulses_and_tone(target):
    n, batch = 16, 4
    x = np.zeros((batch, n), dtype=np.complex64)
    x[0, 0] = 1.0  # impulse at 0: flat spectrum
    x[1, 5] = 1.0  # impulse at 5: a twiddle ramp
    x[2, :] = 1.0  # constant: one bin
    x[3, :] = np.exp(2j * np.pi * 3 * np.arange(n) / n)  # a tone in bin 3
    X = np.zeros(((batch + 1) * n, 2), dtype=np.float32)
    X[: batch * n, 0] = x.real.reshape(-1)
    X[: batch * n, 1] = x.imag.reshape(-1)
    Y = np.zeros((batch * n, 2), dtype=np.float32)
    spmw.build(fft_sdf_of(n, batch), target=target)(X, Y)
    check(x, Y, n, batch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
