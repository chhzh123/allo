# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A fully spatial FFT: key-form links, block streams, and a resident twiddle ROM.

The butterfly's far end is a permutation, so it has no closed-form neighbour --
which is exactly what key form is for: both ends name a shared label and the
compiler completes the pairing.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import float32

FFT_N = 8
S = 3  # log2(FFT_N) stages
HALF = FFT_N // 2  # butterflies per stage

csample = float32[2]  # one complex sample: [re, im]


def bfly_pair(s, b):
    """The two lanes butterfly (s, b) touches."""
    span = 1 << s
    up = (b // span) * (2 * span) + (b % span)
    return up, up + span


def twiddles(n):
    """Elaboration-time Python; numpy is fine here."""
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


class BflyIO(spmw.Interface):
    up_in = spmw.In(csample)
    lo_in = spmw.In(csample)
    up_out = spmw.Out(csample)
    lo_out = spmw.Out(csample)
    tw = spmw.MemIn(float32[HALF, 2])  # same contents at every site


def bfly_links(s, b):
    up, lo = bfly_pair(s, b)
    return {
        BflyIO.up_in: spmw.key(s, up),
        BflyIO.lo_in: spmw.key(s, lo),
        BflyIO.up_out: spmw.key(s + 1, up),
        BflyIO.lo_out: spmw.key(s + 1, lo),
    }


topo = spmw.Topology(BflyIO, grid=(S, HALF), link=bfly_links)


@spmw.unit
def bfly(io: BflyIO, site: spmw.Site):
    s, b = site.rank
    span = 1 << s
    k = b % span * (HALF // span)  # this butterfly's twiddle index
    wr = io.tw[k, 0]
    wi = io.tw[k, 1]
    a = io.up_in.get()
    c = io.lo_in.get()
    tr = wr * c[0] - wi * c[1]  # t = w . x[lo]
    ti = wr * c[1] + wi * c[0]
    u: csample
    l: csample
    u[0] = a[0] + tr  # x'[up] = a + t
    u[1] = a[1] + ti
    l[0] = a[0] - tr  # x'[lo] = a - t
    l[1] = a[1] - ti
    io.up_out.put(u)
    io.lo_out.put(l)


@spmw.fabric
def fft_spatial(X: float32[FFT_N, 2], Y: float32[FFT_N, 2]):
    P = spmw.place(bfly, on=topo)
    tw = spmw.mem(float32[HALF, 2], init=twiddles(FFT_N), layout=spmw.replicate)
    spmw.stationary(tw, at=P.tw)
    # Bit-reversal is no affine factor shuffle: index= as the escape hatch.
    spmw.stream_in(X, into=P.up_in, index=lambda s, b: bitrev(bfly_pair(0, b)[0], S))
    spmw.stream_in(X, into=P.lo_in, index=lambda s, b: bitrev(bfly_pair(0, b)[1], S))
    spmw.gather(Y, from_=P.up_out, index=lambda s, b: bfly_pair(S - 1, b)[0])
    spmw.gather(Y, from_=P.lo_out, index=lambda s, b: bfly_pair(S - 1, b)[1])


def _operands(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random(FFT_N).astype(np.float32) + 1j * rng.random(FFT_N).astype(np.float32)
    X = np.stack([x.real, x.imag], axis=1).astype(np.float32)
    return x, X


def test_elaborates():
    graph = spmw.elaborate(fft_spatial)
    assert [b.kind for b in graph.bindings] == [
        "stationary",
        "stream_in",
        "stream_in",
        "gather",
        "gather",
    ]
    placement = graph.placements[0]
    # Stage 0's inputs have no writer and the last stage's outputs no reader:
    # those are the array's boundary, computed rather than declared.
    assert len(placement.up_in) == HALF
    assert len(placement.lo_in) == HALF
    assert len(placement.up_out) == HALF
    assert len(placement.lo_out) == HALF
    assert all(site[0] == 0 for site in placement.up_in.sites)
    assert all(site[0] == S - 1 for site in placement.up_out.sites)


def test_internal_wires():
    """Every key with both a writer and a reader is one internal lane edge."""
    internal = [c for c in topo.channels.values() if c.writer and c.readers]
    assert len(internal) == (S - 1) * FFT_N


def test_reference_matches_numpy_fft():
    x, X = _operands()
    Y = np.zeros((FFT_N, 2), dtype=np.float32)
    spmw.build(fft_spatial, target="ref")(X, Y)
    got = Y[:, 0] + 1j * Y[:, 1]
    np.testing.assert_allclose(got, np.fft.fft(x), atol=1e-4)


def test_simulator_matches_numpy_fft():
    x, X = _operands()
    Y = np.zeros((FFT_N, 2), dtype=np.float32)
    spmw.build(fft_spatial, target="simulator")(X, Y)
    got = Y[:, 0] + 1j * Y[:, 1]
    np.testing.assert_allclose(got, np.fft.fft(x), atol=1e-4)


@pytest.mark.parametrize("seed", [1, 2])
def test_repeated_inputs(seed):
    x, X = _operands(seed)
    Y = np.zeros((FFT_N, 2), dtype=np.float32)
    spmw.build(fft_spatial, target="simulator")(X, Y)
    got = Y[:, 0] + 1j * Y[:, 1]
    np.testing.assert_allclose(got, np.fft.fft(x), atol=1e-4)


def test_body_is_written_once():
    """Both boundaries share one program; only the wiring differs."""
    text = spmw.source(fft_spatial)
    # One array kernel plus two loaders and two drains.
    assert text.count("df.kernel") == 5
    assert text.count("wr * c[0] - wi * c[1]") <= 3  # one per stage class


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
