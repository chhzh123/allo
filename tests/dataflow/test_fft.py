# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Vectorized 256-point FFT using Allo dataflow, mirroring the architecture of
# gemini-fft.prj/kernel.cpp (N=256, WIDTH=32 complex floats per vector).
#
# Architecture:
#   - Streams carry float32[WIDTH] tokens (real and imaginary separately)
#   - bit_rev_stage:  reads input arrays, outputs bit-reversed chunks to s[0]
#   - intra stage s (0-4, STRIDE=2^s < WIDTH):  butterfly within each WIDTH-chunk
#   - inter stage s (5-7, STRIDE=2^s >= WIDTH): buffer all N, then butterfly
#   - output_stage:  reads s[8], writes result arrays
#
# Twiddle factors are precomputed as numpy constants and embedded inside each
# kernel as local constant arrays, indexed at runtime.
# Bit reversal is computed inline without a LUT (valid for N=256, LOG2_N=8).

import tempfile
from math import cos, sin, pi

import numpy as np
import pytest
import allo
from allo.ir.types import float32, int32, Stream
import allo.dataflow as df
import allo.backend.hls as hls

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N = 256
WIDTH = 32
LOG2_N = 8
NUM_VECS = N // WIDTH  # 8

# Full twiddle LUT: W_N^k = exp(-2*pi*i*k/N) for k=0..N/2-1
full_twr = np.array([cos(-2.0 * pi * k / N) for k in range(N // 2)], dtype=np.float32)
full_twi = np.array([sin(-2.0 * pi * k / N) for k in range(N // 2)], dtype=np.float32)


# ---------------------------------------------------------------------------
# FFT region
# ---------------------------------------------------------------------------
@df.region()
def fft_256(
    inp_re: float32[N],
    inp_im: float32[N],
    out_re: float32[N],
    out_im: float32[N],
):
    # 9 intermediate streams (s[0]..s[8]); each token is WIDTH float32 values
    s_re: Stream[float32[WIDTH], 16][LOG2_N + 1]
    s_im: Stream[float32[WIDTH], 16][LOG2_N + 1]

    # ------------------------------------------------------------------
    # Bit-reversal stage
    # Inline 8-bit bit reversal valid for N=256 (LOG2_N=8).
    # ------------------------------------------------------------------
    @df.kernel(mapping=[1], args=[inp_re, inp_im])
    def bit_rev_stage(local_re: float32[N], local_im: float32[N]):
        buf_re: float32[N] = 0
        buf_im: float32[N] = 0
        for src in range(N):
            rev: int32 = (
                ((src & 1) << 7)
                | ((src & 2) << 5)
                | ((src & 4) << 3)
                | ((src & 8) << 1)
                | ((src & 16) >> 1)
                | ((src & 32) >> 3)
                | ((src & 64) >> 5)
                | ((src & 128) >> 7)
            )
            buf_re[rev] = local_re[src]
            buf_im[rev] = local_im[src]
        for i in range(NUM_VECS):
            chunk_re: float32[WIDTH] = 0
            chunk_im: float32[WIDTH] = 0
            for k in range(WIDTH):
                chunk_re[k] = buf_re[i * WIDTH + k]
                chunk_im[k] = buf_im[i * WIDTH + k]
            s_re[0].put(chunk_re)
            s_im[0].put(chunk_im)

    # ------------------------------------------------------------------
    # Intra-vector stages 0-4  (STRIDE = 2^s < WIDTH)
    # Butterfly pairs are fully contained within one WIDTH-element vector.
    # Twiddle LUT embedded as a local constant array inside each kernel.
    # ------------------------------------------------------------------
    @df.kernel(mapping=[1])
    def intra_0():
        for _i in range(NUM_VECS):
            c_re: float32[WIDTH] = s_re[0].get()
            c_im: float32[WIDTH] = s_im[0].get()
            o_re: float32[WIDTH] = 0
            o_im: float32[WIDTH] = 0
            for k in range(16):  # WIDTH//2, STRIDE=1
                # twiddle = (1, 0) for all k at stage 0
                a_re = c_re[k * 2]
                a_im = c_im[k * 2]
                b_re = c_re[k * 2 + 1]
                b_im = c_im[k * 2 + 1]
                o_re[k * 2] = a_re + b_re
                o_im[k * 2] = a_im + b_im
                o_re[k * 2 + 1] = a_re - b_re
                o_im[k * 2 + 1] = a_im - b_im
            s_re[1].put(o_re)
            s_im[1].put(o_im)

    @df.kernel(mapping=[1])
    def intra_1():
        twr: float32[N // 2] = full_twr
        twi: float32[N // 2] = full_twi
        for _i in range(NUM_VECS):
            c_re: float32[WIDTH] = s_re[1].get()
            c_im: float32[WIDTH] = s_im[1].get()
            o_re: float32[WIDTH] = 0
            o_im: float32[WIDTH] = 0
            for k in range(16):  # STRIDE=2
                il = (k // 2) * 4 + k % 2
                iu = il + 2
                tw_k = (k % 2) * 64  # lb * (N//(2*STRIDE))
                a_re = c_re[il]
                a_im = c_im[il]
                b_re = c_re[iu]
                b_im = c_im[iu]
                tr = twr[tw_k]
                ti = twi[tw_k]
                bw_re: float32 = b_re * tr - b_im * ti
                bw_im: float32 = b_re * ti + b_im * tr
                o_re[il] = a_re + bw_re
                o_im[il] = a_im + bw_im
                o_re[iu] = a_re - bw_re
                o_im[iu] = a_im - bw_im
            s_re[2].put(o_re)
            s_im[2].put(o_im)

    @df.kernel(mapping=[1])
    def intra_2():
        twr: float32[N // 2] = full_twr
        twi: float32[N // 2] = full_twi
        for _i in range(NUM_VECS):
            c_re: float32[WIDTH] = s_re[2].get()
            c_im: float32[WIDTH] = s_im[2].get()
            o_re: float32[WIDTH] = 0
            o_im: float32[WIDTH] = 0
            for k in range(16):  # STRIDE=4
                il = (k // 4) * 8 + k % 4
                iu = il + 4
                tw_k = (k % 4) * 32
                a_re = c_re[il]
                a_im = c_im[il]
                b_re = c_re[iu]
                b_im = c_im[iu]
                tr = twr[tw_k]
                ti = twi[tw_k]
                bw_re: float32 = b_re * tr - b_im * ti
                bw_im: float32 = b_re * ti + b_im * tr
                o_re[il] = a_re + bw_re
                o_im[il] = a_im + bw_im
                o_re[iu] = a_re - bw_re
                o_im[iu] = a_im - bw_im
            s_re[3].put(o_re)
            s_im[3].put(o_im)

    @df.kernel(mapping=[1])
    def intra_3():
        twr: float32[N // 2] = full_twr
        twi: float32[N // 2] = full_twi
        for _i in range(NUM_VECS):
            c_re: float32[WIDTH] = s_re[3].get()
            c_im: float32[WIDTH] = s_im[3].get()
            o_re: float32[WIDTH] = 0
            o_im: float32[WIDTH] = 0
            for k in range(16):  # STRIDE=8
                il = (k // 8) * 16 + k % 8
                iu = il + 8
                tw_k = (k % 8) * 16
                a_re = c_re[il]
                a_im = c_im[il]
                b_re = c_re[iu]
                b_im = c_im[iu]
                tr = twr[tw_k]
                ti = twi[tw_k]
                bw_re: float32 = b_re * tr - b_im * ti
                bw_im: float32 = b_re * ti + b_im * tr
                o_re[il] = a_re + bw_re
                o_im[il] = a_im + bw_im
                o_re[iu] = a_re - bw_re
                o_im[iu] = a_im - bw_im
            s_re[4].put(o_re)
            s_im[4].put(o_im)

    @df.kernel(mapping=[1])
    def intra_4():
        twr: float32[N // 2] = full_twr
        twi: float32[N // 2] = full_twi
        for _i in range(NUM_VECS):
            c_re: float32[WIDTH] = s_re[4].get()
            c_im: float32[WIDTH] = s_im[4].get()
            o_re: float32[WIDTH] = 0
            o_im: float32[WIDTH] = 0
            for k in range(16):  # STRIDE=16
                il = (k // 16) * 32 + k % 16
                iu = il + 16
                tw_k = (k % 16) * 8
                a_re = c_re[il]
                a_im = c_im[il]
                b_re = c_re[iu]
                b_im = c_im[iu]
                tr = twr[tw_k]
                ti = twi[tw_k]
                bw_re: float32 = b_re * tr - b_im * ti
                bw_im: float32 = b_re * ti + b_im * tr
                o_re[il] = a_re + bw_re
                o_im[il] = a_im + bw_im
                o_re[iu] = a_re - bw_re
                o_im[iu] = a_im - bw_im
            s_re[5].put(o_re)
            s_im[5].put(o_im)

    # ------------------------------------------------------------------
    # Inter-vector stages 5-7  (STRIDE = 2^s >= WIDTH)
    # Buffer all N elements, compute N/2 butterflies, then stream out.
    # Separate in/out buffers to avoid read-after-write hazards.
    # ------------------------------------------------------------------
    @df.kernel(mapping=[1])
    def inter_5():
        twr: float32[N // 2] = full_twr
        twi: float32[N // 2] = full_twi
        # STRIDE=32
        in_re: float32[N] = 0
        in_im: float32[N] = 0
        out_re_b: float32[N] = 0
        out_im_b: float32[N] = 0
        for i in range(NUM_VECS):
            chunk_re: float32[WIDTH] = s_re[5].get()
            chunk_im: float32[WIDTH] = s_im[5].get()
            for k in range(WIDTH):
                in_re[i * WIDTH + k] = chunk_re[k]
                in_im[i * WIDTH + k] = chunk_im[k]
        for i in range(NUM_VECS):
            for k in range(16):  # WIDTH//2
                bg = i * 16 + k
                grp = bg // 32
                within = bg % 32
                il = grp * 64 + within
                iu = il + 32
                tw_k = within * 4  # within * (N//(2*32))
                a_re = in_re[il]
                a_im = in_im[il]
                b_re = in_re[iu]
                b_im = in_im[iu]
                tr = twr[tw_k]
                ti = twi[tw_k]
                bw_re: float32 = b_re * tr - b_im * ti
                bw_im: float32 = b_re * ti + b_im * tr
                out_re_b[il] = a_re + bw_re
                out_im_b[il] = a_im + bw_im
                out_re_b[iu] = a_re - bw_re
                out_im_b[iu] = a_im - bw_im
        for i in range(NUM_VECS):
            chunk_re: float32[WIDTH] = 0
            chunk_im: float32[WIDTH] = 0
            for k in range(WIDTH):
                chunk_re[k] = out_re_b[i * WIDTH + k]
                chunk_im[k] = out_im_b[i * WIDTH + k]
            s_re[6].put(chunk_re)
            s_im[6].put(chunk_im)

    @df.kernel(mapping=[1])
    def inter_6():
        twr: float32[N // 2] = full_twr
        twi: float32[N // 2] = full_twi
        # STRIDE=64
        in_re: float32[N] = 0
        in_im: float32[N] = 0
        out_re_b: float32[N] = 0
        out_im_b: float32[N] = 0
        for i in range(NUM_VECS):
            chunk_re: float32[WIDTH] = s_re[6].get()
            chunk_im: float32[WIDTH] = s_im[6].get()
            for k in range(WIDTH):
                in_re[i * WIDTH + k] = chunk_re[k]
                in_im[i * WIDTH + k] = chunk_im[k]
        for i in range(NUM_VECS):
            for k in range(16):
                bg = i * 16 + k
                grp = bg // 64
                within = bg % 64
                il = grp * 128 + within
                iu = il + 64
                tw_k = within * 2  # within * (N//(2*64))
                a_re = in_re[il]
                a_im = in_im[il]
                b_re = in_re[iu]
                b_im = in_im[iu]
                tr = twr[tw_k]
                ti = twi[tw_k]
                bw_re: float32 = b_re * tr - b_im * ti
                bw_im: float32 = b_re * ti + b_im * tr
                out_re_b[il] = a_re + bw_re
                out_im_b[il] = a_im + bw_im
                out_re_b[iu] = a_re - bw_re
                out_im_b[iu] = a_im - bw_im
        for i in range(NUM_VECS):
            chunk_re: float32[WIDTH] = 0
            chunk_im: float32[WIDTH] = 0
            for k in range(WIDTH):
                chunk_re[k] = out_re_b[i * WIDTH + k]
                chunk_im[k] = out_im_b[i * WIDTH + k]
            s_re[7].put(chunk_re)
            s_im[7].put(chunk_im)

    @df.kernel(mapping=[1])
    def inter_7():
        twr: float32[N // 2] = full_twr
        twi: float32[N // 2] = full_twi
        # STRIDE=128
        in_re: float32[N] = 0
        in_im: float32[N] = 0
        out_re_b: float32[N] = 0
        out_im_b: float32[N] = 0
        for i in range(NUM_VECS):
            chunk_re: float32[WIDTH] = s_re[7].get()
            chunk_im: float32[WIDTH] = s_im[7].get()
            for k in range(WIDTH):
                in_re[i * WIDTH + k] = chunk_re[k]
                in_im[i * WIDTH + k] = chunk_im[k]
        for i in range(NUM_VECS):
            for k in range(16):
                bg = i * 16 + k
                grp = bg // 128
                within = bg % 128
                il = grp * 256 + within
                iu = il + 128
                tw_k = within  # within * (N//(2*128)) = within
                a_re = in_re[il]
                a_im = in_im[il]
                b_re = in_re[iu]
                b_im = in_im[iu]
                tr = twr[tw_k]
                ti = twi[tw_k]
                bw_re: float32 = b_re * tr - b_im * ti
                bw_im: float32 = b_re * ti + b_im * tr
                out_re_b[il] = a_re + bw_re
                out_im_b[il] = a_im + bw_im
                out_re_b[iu] = a_re - bw_re
                out_im_b[iu] = a_im - bw_im
        for i in range(NUM_VECS):
            chunk_re: float32[WIDTH] = 0
            chunk_im: float32[WIDTH] = 0
            for k in range(WIDTH):
                chunk_re[k] = out_re_b[i * WIDTH + k]
                chunk_im[k] = out_im_b[i * WIDTH + k]
            s_re[8].put(chunk_re)
            s_im[8].put(chunk_im)

    # ------------------------------------------------------------------
    # Output stage
    # ------------------------------------------------------------------
    @df.kernel(mapping=[1], args=[out_re, out_im])
    def output_stage(local_re: float32[N], local_im: float32[N]):
        for i in range(NUM_VECS):
            chunk_re: float32[WIDTH] = s_re[8].get()
            chunk_im: float32[WIDTH] = s_im[8].get()
            for k in range(WIDTH):
                local_re[i * WIDTH + k] = chunk_re[k]
                local_im[i * WIDTH + k] = chunk_im[k]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_fft_256():
    np.random.seed(42)
    inp_re = np.random.rand(N).astype(np.float32)
    inp_im = np.zeros(N, dtype=np.float32)
    out_re = np.zeros(N, dtype=np.float32)
    out_im = np.zeros(N, dtype=np.float32)

    sim_mod = df.build(fft_256, target="simulator")
    sim_mod(inp_re, inp_im, out_re, out_im)

    ref = np.fft.fft(inp_re + 1j * inp_im)
    np.testing.assert_allclose(out_re, ref.real.astype(np.float32), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_im, ref.imag.astype(np.float32), rtol=1e-4, atol=1e-4)
    print("FFT-256 Simulator Passed!")


def test_fft_256_hls_codegen():
    """Verify HLS code is generated with the expected structure."""
    mod = df.build(fft_256, target="vitis_hls")
    code = mod.hls_code
    assert "hls::vector" in code, "Expected vectorized stream type in HLS code"
    assert "hls::stream" in code
    assert "bit_rev_stage" in code
    assert "inter_7" in code
    print("FFT-256 HLS Codegen Passed!")


def test_fft_256_csyn():
    """Run C-synthesis via Vitis HLS."""
    if not hls.is_available("vitis_hls"):
        pytest.skip("Vitis HLS not available")
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = df.build(fft_256, target="vitis_hls", mode="csyn", project=tmpdir)
        mod()
    print("FFT-256 CSyn Passed!")


if __name__ == "__main__":
    test_fft_256()
    test_fft_256_hls_codegen()
