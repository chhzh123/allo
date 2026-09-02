# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A fully spatial FFT: key-form links, block streams, and a resident twiddle ROM.

The butterfly's far end is a permutation, so it has no closed-form neighbour --
which is exactly what key form is for: both ends name a shared label and the
compiler completes the pairing.
"""

import tempfile

import numpy as np
import pytest

import allo.backend.hls as hls
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


# -- the streaming form ------------------------------------------------------
#
# The design above computes one transform per pass: every butterfly reads one
# sample pair, does its arithmetic and stops. HLS reports latency 65, interval
# 66 -- there is no loop, so there is nothing to pipeline, and the float
# latencies are paid in full on every transform.
#
# A butterfly is feed-forward: iteration i does not depend on i-1. So a batch
# turns those latencies into pipeline depth. `BATCH` transforms stream through
# the same array back to back, and the interval that matters becomes the one
# per sample pair rather than per transform.

BATCH = 64


class StreamIO(spmw.Interface):
    up_in = spmw.In(csample)
    lo_in = spmw.In(csample)
    up_out = spmw.Out(csample)
    lo_out = spmw.Out(csample)
    tw = spmw.MemIn(float32[HALF, 2])


def stream_links(s, b):
    up, lo = bfly_pair(s, b)
    return {
        StreamIO.up_in: spmw.key(s, up),
        StreamIO.lo_in: spmw.key(s, lo),
        StreamIO.up_out: spmw.key(s + 1, up),
        StreamIO.lo_out: spmw.key(s + 1, lo),
    }


stream_topo = spmw.Topology(StreamIO, grid=(S, HALF), link=stream_links)


@spmw.unit
def bfly_stream(io: StreamIO, site: spmw.Site):
    s, b = site.rank
    span = 1 << s
    k = b % span * (HALF // span)
    wr = io.tw[k, 0]
    wi = io.tw[k, 1]
    for _n in range(BATCH):
        a = io.up_in.get()
        c = io.lo_in.get()
        tr = wr * c[0] - wi * c[1]
        ti = wr * c[1] + wi * c[0]
        u: csample
        l: csample
        u[0] = a[0] + tr
        u[1] = a[1] + ti
        l[0] = a[0] - tr
        l[1] = a[1] - ti
        io.up_out.put(u)
        io.lo_out.put(l)


@spmw.fabric
def fft_stream(X: float32[BATCH, FFT_N, 2], Y: float32[BATCH, FFT_N, 2]):
    P = spmw.place(bfly_stream, on=stream_topo)
    tw = spmw.mem(float32[HALF, 2], init=twiddles(FFT_N), layout=spmw.replicate)
    spmw.stationary(tw, at=P.tw)
    # Affine, where the one-shot design needed a lambda. At stage 0 butterfly b
    # touches lanes 2b and 2b+1; at the last stage it produces lanes b and
    # b+HALF. The only part that was not affine is the bit-reversal, and that
    # is a permutation of the *input*, not of the network -- so the host hands
    # X over in bit-reversed order, which is where FFT hardware puts it anyway.
    # `...` is the batch axis: one transform per step.
    _stage, b = P.axes
    spmw.stream_in(X, into=P.up_in, index=(..., 2 * b))
    spmw.stream_in(X, into=P.lo_in, index=(..., 2 * b + 1))
    spmw.gather(Y, from_=P.up_out, index=(..., b))
    spmw.gather(Y, from_=P.lo_out, index=(..., b + HALF))


def bitrev_perm(n, bits):
    """The order the streaming form wants its input in."""
    return [bitrev(i, bits) for i in range(n)]


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


@pytest.mark.skipif(not hls.is_available("vitis_hls"), reason="vitis_hls not on PATH")
def test_hls_csim_matches_numpy_fft():
    """Block-carrying streams and a resident ROM, through the HLS path."""
    x, X = _operands()
    Y = np.zeros((FFT_N, 2), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        spmw.build(fft_spatial, target="vitis_hls", mode="csim", project=tmpdir)(X, Y)
    got = Y[:, 0] + 1j * Y[:, 1]
    np.testing.assert_allclose(got, np.fft.fft(x), atol=1e-4)


def fft_stream_of(n, batch, name=None):
    """The streaming butterfly network at any power-of-two size.

    `FFT_N` above is 8 so the tests stay readable. The reference this is
    measured against is N=256, which is 8 stages of 128 butterflies -- 1,024
    units, the same order as the 32x32 matrix array.
    """
    stages = n.bit_length() - 1
    half = n // 2
    sample = float32[2]

    def pair(s, b):
        span = 1 << s
        up = (b // span) * (2 * span) + (b % span)
        return up, up + span

    class IO(spmw.Interface):
        up_in = spmw.In(sample)
        lo_in = spmw.In(sample)
        up_out = spmw.Out(sample)
        lo_out = spmw.Out(sample)
        tw = spmw.MemIn(float32[half, 2])

    def links(s, b):
        up, lo = pair(s, b)
        return {
            IO.up_in: spmw.key(s, up),
            IO.lo_in: spmw.key(s, lo),
            IO.up_out: spmw.key(s + 1, up),
            IO.lo_out: spmw.key(s + 1, lo),
        }

    topology = spmw.Topology(IO, grid=(stages, half), link=links)

    @spmw.unit
    def butterfly(io: IO, site: spmw.Site):
        s, b = site.rank
        span = 1 << s
        k = b % span * (half // span)
        wr = io.tw[k, 0]
        wi = io.tw[k, 1]
        for _n in range(batch):
            a = io.up_in.get()
            c = io.lo_in.get()
            tr = wr * c[0] - wi * c[1]
            ti = wr * c[1] + wi * c[0]
            u: sample
            l: sample
            u[0] = a[0] + tr
            u[1] = a[1] + ti
            l[0] = a[0] - tr
            l[1] = a[1] - ti
            io.up_out.put(u)
            io.lo_out.put(l)

    @spmw.fabric
    def fabric(X: float32[batch, n, 2], Y: float32[batch, n, 2]):
        P = spmw.place(butterfly, on=topology)
        tw = spmw.mem(float32[half, 2], init=twiddles(n), layout=spmw.replicate)
        spmw.stationary(tw, at=P.tw)
        _stage, b = P.axes
        spmw.stream_in(X, into=P.up_in, index=(..., 2 * b))
        spmw.stream_in(X, into=P.lo_in, index=(..., 2 * b + 1))
        spmw.gather(Y, from_=P.up_out, index=(..., b))
        spmw.gather(Y, from_=P.lo_out, index=(..., b + half))

    if name:
        fabric.__name__ = name
    return fabric


def _stream_operands(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((BATCH, FFT_N)) + 1j * rng.random((BATCH, FFT_N))
    xb = x[:, bitrev_perm(FFT_N, S)]
    # Contiguous: the permutation above makes a strided view, and the LLVM
    # path takes memrefs by pointer.
    X = np.ascontiguousarray(np.stack([xb.real, xb.imag], axis=-1).astype(np.float32))
    return x, X


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_streaming_fft_matches_numpy(target):
    """A batch of transforms through the same butterfly network."""
    x, X = _stream_operands()
    Y = np.zeros((BATCH, FFT_N, 2), dtype=np.float32)
    spmw.build(fft_stream, target=target)(X, Y)
    got = Y[..., 0] + 1j * Y[..., 1]
    np.testing.assert_allclose(got, np.fft.fft(x, axis=-1), atol=1e-4)


def test_the_streaming_butterfly_has_a_loop_to_pipeline():
    """The whole point of the batch, asserted on the emitted body.

    The one-shot butterfly reads one sample pair and stops, so HLS has nothing
    to pipeline and charges the float latencies per transform -- measured at
    latency 65, interval 66. The streaming one wraps the same arithmetic in a
    loop over the batch, which is what lets the latencies become depth instead
    of throughput. Without the loop there is no II to speak of.
    """
    one_shot = spmw.source(fft_spatial)
    streaming = spmw.source(fft_stream)
    # The arithmetic is the same in both.
    assert "wr * c[0] - wi * c[1]" in one_shot
    assert "wr * c[0] - wi * c[1]" in streaming
    # Only the streaming form loops over a batch.
    assert f"range({BATCH})" in streaming
    assert f"range({BATCH})" not in one_shot


def test_the_streaming_index_is_affine():
    """No lambda in the streaming bindings, which is why `...` can be used.

    Bit-reversal is the one non-affine part of an FFT's wiring, and it is a
    permutation of the input rather than of the network -- so it moves to the
    host and every remaining index is an axis expression. `bitrev_perm` is that
    permutation, and the tests above feed it.
    """
    graph = spmw.elaborate(fft_stream)
    streamed = [b for b in graph.bindings if b.kind in ("stream_in", "gather")]
    assert len(streamed) == 4
    for binding in streamed:
        assert not binding.imap.is_lambda, binding.kind
        assert binding.imap.has_time, "the batch axis should be the streamed one"


def test_bit_reversal_is_a_permutation():
    """It has to be, or the host would be dropping samples."""
    perm = bitrev_perm(FFT_N, S)
    assert sorted(perm) == list(range(FFT_N))
    assert perm != list(range(FFT_N)), "a trivial permutation would test nothing"


def test_body_is_written_once():
    """Both boundaries share one program; only the wiring differs."""
    text = spmw.source(fft_spatial)
    # One array kernel plus two loaders and two drains.
    assert text.count("df.kernel") == 5
    assert text.count("wr * c[0] - wi * c[1]") <= 3  # one per stage class


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
