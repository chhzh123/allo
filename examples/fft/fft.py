# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HP-FFT: High-Performance Streaming FFT

256-point FFT with configurable unroll factor (UF).

Architecture:
- Intra-chunk stages: butterflies where span < CHUNK (true streaming)
- Cross-chunk stages: butterflies where span >= CHUNK (3-phase pattern)

The number of intra-chunk vs cross-chunk stages depends on UF:
- UF=1: CHUNK=2, 1 intra-chunk stage (stage 0), 7 cross-chunk stages
- UF=2: CHUNK=4, 2 intra-chunk stages (0-1), 6 cross-chunk stages
- UF=4: CHUNK=8, 3 intra-chunk stages (0-2), 5 cross-chunk stages

Expected performance: ~597 cycles latency, ~268 cycle interval (for UF=2)
"""

import os
import sys
import argparse
from math import log2, cos, sin, pi

import allo
from allo.ir.types import float32, int32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np


def bit_reverse(x, bits):
    result = 0
    for i in range(bits):
        if x & (1 << i):
            result |= 1 << (bits - 1 - i)
    return result


def create_fft(N: int, UF: int):
    """Create FFT dataflow for N-point FFT with given unroll factor.

    Only UF=2 is currently supported since the stage logic is specialized.
    Other UF values would require different stage configurations.
    """
    if UF != 2:
        raise ValueError(f"Only UF=2 is currently supported. Got UF={UF}")

    LOG2_N = int(log2(N))
    HALF_N = N // 2
    CHUNK = UF * 2  # 4 elements per chunk for UF=2
    NUM_CHUNKS = N // CHUNK  # 64 chunks for N=256

    np_bit_rev = np.array([bit_reverse(i, LOG2_N) for i in range(N)], dtype=np.int32)

    def make_twiddles(stage):
        span = 1 << stage
        tw_r = np.zeros(span, dtype=np.float32)
        tw_i = np.zeros(span, dtype=np.float32)
        for k in range(span):
            angle = -2.0 * pi * k / (2 * span)
            tw_r[k] = cos(angle)
            tw_i[k] = sin(angle)
        return tw_r, tw_i

    tw0_r, tw0_i = make_twiddles(0)  # span=1
    tw1_r, tw1_i = make_twiddles(1)  # span=2
    tw2_r, tw2_i = make_twiddles(2)  # span=4
    tw3_r, tw3_i = make_twiddles(3)  # span=8
    tw4_r, tw4_i = make_twiddles(4)  # span=16
    tw5_r, tw5_i = make_twiddles(5)  # span=32
    tw6_r, tw6_i = make_twiddles(6)  # span=64
    tw7_r, tw7_i = make_twiddles(7)  # span=128

    # Concatenate all twiddles for stages 2-7 into one array with offsets
    # Offsets: stage2=0, stage3=4, stage4=12, stage5=28, stage6=60, stage7=124
    all_tw_r = np.concatenate([tw2_r, tw3_r, tw4_r, tw5_r, tw6_r, tw7_r])
    all_tw_i = np.concatenate([tw2_i, tw3_i, tw4_i, tw5_i, tw6_i, tw7_i])

    @df.region()
    def top(
        inp_real: float32[N],
        inp_imag: float32[N],
        out_real: float32[N],
        out_imag: float32[N],
    ):
        # Streams between stages
        s0_r: Stream[float32[CHUNK], 4]
        s0_i: Stream[float32[CHUNK], 4]
        s1_r: Stream[float32[CHUNK], 4]
        s1_i: Stream[float32[CHUNK], 4]
        s2_r: Stream[float32[CHUNK], 4]
        s2_i: Stream[float32[CHUNK], 4]
        s3_r: Stream[float32[CHUNK], 4]
        s3_i: Stream[float32[CHUNK], 4]
        s4_r: Stream[float32[CHUNK], 4]
        s4_i: Stream[float32[CHUNK], 4]
        s5_r: Stream[float32[CHUNK], 4]
        s5_i: Stream[float32[CHUNK], 4]
        s6_r: Stream[float32[CHUNK], 4]
        s6_i: Stream[float32[CHUNK], 4]
        s7_r: Stream[float32[CHUNK], 4]
        s7_i: Stream[float32[CHUNK], 4]
        s8_r: Stream[float32[CHUNK], 4]
        s8_i: Stream[float32[CHUNK], 4]

        # Input loader with bit-reversal
        @df.kernel(mapping=[1], args=[inp_real, inp_imag])
        def input_loader(local_inp_real: float32[N], local_inp_imag: float32[N]):
            bit_rev: int32[N] = np_bit_rev
            for c in range(NUM_CHUNKS):
                buf_r: float32[CHUNK]
                buf_i: float32[CHUNK]
                for u in range(CHUNK):
                    src: int32 = c * CHUNK + u
                    rev: int32 = bit_rev[src]
                    buf_r[u] = local_inp_real[rev]
                    buf_i[u] = local_inp_imag[rev]
                s0_r.put(buf_r)
                s0_i.put(buf_i)

        # Stage 0: span=1, butterflies (0,1) and (2,3) within chunk
        @df.kernel(mapping=[1])
        def stage0():
            tw_r: float32[1] = tw0_r
            tw_i: float32[1] = tw0_i
            for c in range(NUM_CHUNKS):
                ir: float32[CHUNK] = s0_r.get()
                ii: float32[CHUNK] = s0_i.get()
                or_out: float32[CHUNK]
                oi_out: float32[CHUNK]
                for u in range(UF):
                    u_idx: int32 = u * 2
                    l_idx: int32 = u_idx + 1
                    wr: float32 = tw_r[0]
                    wi: float32 = tw_i[0]
                    bwr: float32 = ir[l_idx] * wr - ii[l_idx] * wi
                    bwi: float32 = ir[l_idx] * wi + ii[l_idx] * wr
                    or_out[u_idx] = ir[u_idx] + bwr
                    oi_out[u_idx] = ii[u_idx] + bwi
                    or_out[l_idx] = ir[u_idx] - bwr
                    oi_out[l_idx] = ii[u_idx] - bwi
                s1_r.put(or_out)
                s1_i.put(oi_out)

        # Stage 1: span=2, butterflies (0,2) and (1,3) within chunk
        @df.kernel(mapping=[1])
        def stage1():
            tw_r: float32[2] = tw1_r
            tw_i: float32[2] = tw1_i
            for c in range(NUM_CHUNKS):
                ir: float32[CHUNK] = s1_r.get()
                ii: float32[CHUNK] = s1_i.get()
                or_out: float32[CHUNK]
                oi_out: float32[CHUNK]
                for u in range(UF):
                    u_idx: int32 = u
                    l_idx: int32 = u + UF
                    wr: float32 = tw_r[u]
                    wi: float32 = tw_i[u]
                    bwr: float32 = ir[l_idx] * wr - ii[l_idx] * wi
                    bwi: float32 = ir[l_idx] * wi + ii[l_idx] * wr
                    or_out[u_idx] = ir[u_idx] + bwr
                    oi_out[u_idx] = ii[u_idx] + bwi
                    or_out[l_idx] = ir[u_idx] - bwr
                    oi_out[l_idx] = ii[u_idx] - bwi
                s2_r.put(or_out)
                s2_i.put(oi_out)

        # =================================================================
        # STAGES 2-7: 3-phase pattern using mapping with get_pid()
        #
        # Each stage is unrolled at compile time using meta_if on get_pid()
        # to select the appropriate streams and twiddle factors.
        # =================================================================

        @df.kernel(mapping=[6])
        def cross_chunk_stages():
            pid = df.get_pid()
            all_twiddles_r: float32[252] = all_tw_r
            all_twiddles_i: float32[252] = all_tw_i
            buf_r: float32[N]
            buf_i: float32[N]
            res_r: float32[N]
            res_i: float32[N]

            # Stage 2: span=4, shift=2
            with allo.meta_if(pid == 0):
                for c in range(NUM_CHUNKS):
                    tmp_r: float32[CHUNK] = s2_r.get()
                    tmp_i: float32[CHUNK] = s2_i.get()
                    for u in range(CHUNK):
                        idx: int32 = c * CHUNK + u
                        buf_r[idx] = tmp_r[u]
                        buf_i[idx] = tmp_i[u]
                for b in range(HALF_N):
                    group: int32 = b >> 2
                    pos: int32 = b & 3
                    u_idx: int32 = group * 8 + pos
                    l_idx: int32 = u_idx + 4
                    wr: float32 = all_twiddles_r[pos]  # offset 0
                    wi: float32 = all_twiddles_i[pos]
                    bwr: float32 = buf_r[l_idx] * wr - buf_i[l_idx] * wi
                    bwi: float32 = buf_r[l_idx] * wi + buf_i[l_idx] * wr
                    res_r[u_idx] = buf_r[u_idx] + bwr
                    res_i[u_idx] = buf_i[u_idx] + bwi
                    res_r[l_idx] = buf_r[u_idx] - bwr
                    res_i[l_idx] = buf_i[u_idx] - bwi
                for c in range(NUM_CHUNKS):
                    out_r: float32[CHUNK]
                    out_i: float32[CHUNK]
                    for u in range(CHUNK):
                        idx: int32 = c * CHUNK + u
                        out_r[u] = res_r[idx]
                        out_i[u] = res_i[idx]
                    s3_r.put(out_r)
                    s3_i.put(out_i)

            # Stage 3: span=8, shift=3
            with allo.meta_if(pid == 1):
                for c in range(NUM_CHUNKS):
                    tmp_r: float32[CHUNK] = s3_r.get()
                    tmp_i: float32[CHUNK] = s3_i.get()
                    for u in range(CHUNK):
                        idx: int32 = c * CHUNK + u
                        buf_r[idx] = tmp_r[u]
                        buf_i[idx] = tmp_i[u]
                for b in range(HALF_N):
                    group: int32 = b >> 3
                    pos: int32 = b & 7
                    u_idx: int32 = group * 16 + pos
                    l_idx: int32 = u_idx + 8
                    wr: float32 = all_twiddles_r[4 + pos]  # offset 4
                    wi: float32 = all_twiddles_i[4 + pos]
                    bwr: float32 = buf_r[l_idx] * wr - buf_i[l_idx] * wi
                    bwi: float32 = buf_r[l_idx] * wi + buf_i[l_idx] * wr
                    res_r[u_idx] = buf_r[u_idx] + bwr
                    res_i[u_idx] = buf_i[u_idx] + bwi
                    res_r[l_idx] = buf_r[u_idx] - bwr
                    res_i[l_idx] = buf_i[u_idx] - bwi
                for c in range(NUM_CHUNKS):
                    out_r: float32[CHUNK]
                    out_i: float32[CHUNK]
                    for u in range(CHUNK):
                        idx: int32 = c * CHUNK + u
                        out_r[u] = res_r[idx]
                        out_i[u] = res_i[idx]
                    s4_r.put(out_r)
                    s4_i.put(out_i)

            # Stage 4: span=16, shift=4
            with allo.meta_if(pid == 2):
                for c in range(NUM_CHUNKS):
                    tmp_r: float32[CHUNK] = s4_r.get()
                    tmp_i: float32[CHUNK] = s4_i.get()
                    for u in range(CHUNK):
                        idx: int32 = c * CHUNK + u
                        buf_r[idx] = tmp_r[u]
                        buf_i[idx] = tmp_i[u]
                for b in range(HALF_N):
                    group: int32 = b >> 4
                    pos: int32 = b & 15
                    u_idx: int32 = group * 32 + pos
                    l_idx: int32 = u_idx + 16
                    wr: float32 = all_twiddles_r[12 + pos]  # offset 4+8=12
                    wi: float32 = all_twiddles_i[12 + pos]
                    bwr: float32 = buf_r[l_idx] * wr - buf_i[l_idx] * wi
                    bwi: float32 = buf_r[l_idx] * wi + buf_i[l_idx] * wr
                    res_r[u_idx] = buf_r[u_idx] + bwr
                    res_i[u_idx] = buf_i[u_idx] + bwi
                    res_r[l_idx] = buf_r[u_idx] - bwr
                    res_i[l_idx] = buf_i[u_idx] - bwi
                for c in range(NUM_CHUNKS):
                    out_r: float32[CHUNK]
                    out_i: float32[CHUNK]
                    for u in range(CHUNK):
                        idx: int32 = c * CHUNK + u
                        out_r[u] = res_r[idx]
                        out_i[u] = res_i[idx]
                    s5_r.put(out_r)
                    s5_i.put(out_i)

            # Stage 5: span=32, shift=5
            with allo.meta_if(pid == 3):
                for c in range(NUM_CHUNKS):
                    tmp_r: float32[CHUNK] = s5_r.get()
                    tmp_i: float32[CHUNK] = s5_i.get()
                    for u in range(CHUNK):
                        idx: int32 = c * CHUNK + u
                        buf_r[idx] = tmp_r[u]
                        buf_i[idx] = tmp_i[u]
                for b in range(HALF_N):
                    group: int32 = b >> 5
                    pos: int32 = b & 31
                    u_idx: int32 = group * 64 + pos
                    l_idx: int32 = u_idx + 32
                    wr: float32 = all_twiddles_r[28 + pos]  # offset 4+8+16=28
                    wi: float32 = all_twiddles_i[28 + pos]
                    bwr: float32 = buf_r[l_idx] * wr - buf_i[l_idx] * wi
                    bwi: float32 = buf_r[l_idx] * wi + buf_i[l_idx] * wr
                    res_r[u_idx] = buf_r[u_idx] + bwr
                    res_i[u_idx] = buf_i[u_idx] + bwi
                    res_r[l_idx] = buf_r[u_idx] - bwr
                    res_i[l_idx] = buf_i[u_idx] - bwi
                for c in range(NUM_CHUNKS):
                    out_r: float32[CHUNK]
                    out_i: float32[CHUNK]
                    for u in range(CHUNK):
                        idx: int32 = c * CHUNK + u
                        out_r[u] = res_r[idx]
                        out_i[u] = res_i[idx]
                    s6_r.put(out_r)
                    s6_i.put(out_i)

            # Stage 6: span=64, shift=6
            with allo.meta_if(pid == 4):
                for c in range(NUM_CHUNKS):
                    tmp_r: float32[CHUNK] = s6_r.get()
                    tmp_i: float32[CHUNK] = s6_i.get()
                    for u in range(CHUNK):
                        idx: int32 = c * CHUNK + u
                        buf_r[idx] = tmp_r[u]
                        buf_i[idx] = tmp_i[u]
                for b in range(HALF_N):
                    group: int32 = b >> 6
                    pos: int32 = b & 63
                    u_idx: int32 = group * 128 + pos
                    l_idx: int32 = u_idx + 64
                    wr: float32 = all_twiddles_r[60 + pos]  # offset 28+32=60
                    wi: float32 = all_twiddles_i[60 + pos]
                    bwr: float32 = buf_r[l_idx] * wr - buf_i[l_idx] * wi
                    bwi: float32 = buf_r[l_idx] * wi + buf_i[l_idx] * wr
                    res_r[u_idx] = buf_r[u_idx] + bwr
                    res_i[u_idx] = buf_i[u_idx] + bwi
                    res_r[l_idx] = buf_r[u_idx] - bwr
                    res_i[l_idx] = buf_i[u_idx] - bwi
                for c in range(NUM_CHUNKS):
                    out_r: float32[CHUNK]
                    out_i: float32[CHUNK]
                    for u in range(CHUNK):
                        idx: int32 = c * CHUNK + u
                        out_r[u] = res_r[idx]
                        out_i[u] = res_i[idx]
                    s7_r.put(out_r)
                    s7_i.put(out_i)

            # Stage 7: span=128, shift=7
            with allo.meta_if(pid == 5):
                for c in range(NUM_CHUNKS):
                    tmp_r: float32[CHUNK] = s7_r.get()
                    tmp_i: float32[CHUNK] = s7_i.get()
                    for u in range(CHUNK):
                        idx: int32 = c * CHUNK + u
                        buf_r[idx] = tmp_r[u]
                        buf_i[idx] = tmp_i[u]
                for b in range(HALF_N):
                    pos: int32 = b
                    u_idx: int32 = pos
                    l_idx: int32 = pos + 128
                    wr: float32 = all_twiddles_r[124 + pos]  # offset 60+64=124
                    wi: float32 = all_twiddles_i[124 + pos]
                    bwr: float32 = buf_r[l_idx] * wr - buf_i[l_idx] * wi
                    bwi: float32 = buf_r[l_idx] * wi + buf_i[l_idx] * wr
                    res_r[u_idx] = buf_r[u_idx] + bwr
                    res_i[u_idx] = buf_i[u_idx] + bwi
                    res_r[l_idx] = buf_r[u_idx] - bwr
                    res_i[l_idx] = buf_i[u_idx] - bwi
                for c in range(NUM_CHUNKS):
                    out_r: float32[CHUNK]
                    out_i: float32[CHUNK]
                    for u in range(CHUNK):
                        idx: int32 = c * CHUNK + u
                        out_r[u] = res_r[idx]
                        out_i[u] = res_i[idx]
                    s8_r.put(out_r)
                    s8_i.put(out_i)

        @df.kernel(mapping=[1], args=[out_real, out_imag])
        def output_store(local_out_real: float32[N], local_out_imag: float32[N]):
            for c in range(NUM_CHUNKS):
                tmp_r: float32[CHUNK] = s8_r.get()
                tmp_i: float32[CHUNK] = s8_i.get()
                for u in range(CHUNK):
                    idx: int32 = c * CHUNK + u
                    local_out_real[idx] = tmp_r[u]
                    local_out_imag[idx] = tmp_i[u]

    return top, N, UF


def test_fft(mod, N, UF):
    np.random.seed(42)
    inp_real = np.random.rand(N).astype(np.float32)
    inp_imag = np.zeros(N, dtype=np.float32)
    out_real = np.zeros(N, dtype=np.float32)
    out_imag = np.zeros(N, dtype=np.float32)

    LOG2_N = int(log2(N))
    print(f"HP-FFT {N}-point ({LOG2_N} stages, UF={UF})")

    mod(inp_real, inp_imag, out_real, out_imag)

    ref = np.fft.fft(inp_real + 1j * inp_imag)
    max_diff_real = np.max(np.abs(out_real - ref.real))
    max_diff_imag = np.max(np.abs(out_imag - ref.imag))

    print(f"Max diff: real={max_diff_real:.6e}, imag={max_diff_imag:.6e}")

    try:
        np.testing.assert_allclose(out_real, ref.real, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out_imag, ref.imag, rtol=1e-4, atol=1e-4)
        print("✅ PASSED")
        return True
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
        raise RuntimeError("Test failed")


def main():
    parser = argparse.ArgumentParser(description="HP-FFT")
    parser.add_argument("--uf", type=int, default=2)
    parser.add_argument("--sim-only", action="store_true")
    args = parser.parse_args()

    N, UF = 256, args.uf
    os.environ["OMP_NUM_THREADS"] = str(max(64, N * 2))

    top, _, _ = create_fft(N, UF)
    mod = df.build(top, target="simulator")
    test_fft(mod, N, UF)

    if args.sim_only:
        return

    mod = df.build(
        top,
        target="vitis_hls",
        mode="hw_emu",
        project=f"fft_{N}_uf{UF}.prj",
        configs={"num_output_args": 2, "frequency": 250},
        wrap_io=False,
    )
    test_fft(mod, N, UF)

    del os.environ["OMP_NUM_THREADS"]


if __name__ == "__main__":
    main()
