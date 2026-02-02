# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HP-FFT: High-Performance Streaming FFT (Merged Stages Version)

256-point FFT with UF=2, using stream arrays to merge stages 2-7.
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
    """Create FFT dataflow for N-point FFT with given unroll factor."""
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

    # Concatenate all twiddles for stages 2-7 into one array
    # Stage 2: span=4, offset=0
    # Stage 3: span=8, offset=4
    # Stage 4: span=16, offset=12
    # Stage 5: span=32, offset=28
    # Stage 6: span=64, offset=60
    # Stage 7: span=128, offset=124
    all_tw_r = np.concatenate([make_twiddles(s)[0] for s in range(2, 8)])
    all_tw_i = np.concatenate([make_twiddles(s)[1] for s in range(2, 8)])

    # Precompute offsets: offset[k] = sum of spans from stage 2 to stage (2+k-1)
    # For pid=0 (stage 2): offset=0
    # For pid=1 (stage 3): offset=4
    # For pid=2 (stage 4): offset=4+8=12
    # etc.
    np_offsets = np.array([0, 4, 12, 28, 60, 124], dtype=np.int32)
    np_spans = np.array([4, 8, 16, 32, 64, 128], dtype=np.int32)

    @df.region()
    def top(
        inp_real: float32[N],
        inp_imag: float32[N],
        out_real: float32[N],
        out_imag: float32[N],
    ):
        # Stream arrays for cross-chunk stages (indices 0-6 for s2-s8)
        # s_r[0]/s_i[0] = input to stage 2 (from stage 1)
        # s_r[6]/s_i[6] = output of stage 7 (to output_store)
        s_r: Stream[float32[CHUNK], 4][7]
        s_i: Stream[float32[CHUNK], 4][7]

        # Streams for intra-chunk stages
        s0_r: Stream[float32[CHUNK], 4]
        s0_i: Stream[float32[CHUNK], 4]
        s1_r: Stream[float32[CHUNK], 4]
        s1_i: Stream[float32[CHUNK], 4]

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
                # Output to first stream array element (input to stage 2)
                s_r[0].put(or_out)
                s_i[0].put(oi_out)

        # =================================================================
        # STAGES 2-7: Merged into one kernel using stream arrays
        #
        # Each of the 6 instances reads from s_r[pid]/s_i[pid]
        # and writes to s_r[pid+1]/s_i[pid+1]
        # =================================================================
        @df.kernel(mapping=[6])
        def cross_chunk_stage():
            pid = df.get_pid()
            all_twiddles_r: float32[252] = all_tw_r
            all_twiddles_i: float32[252] = all_tw_i
            offsets: int32[6] = np_offsets
            spans: int32[6] = np_spans

            buf_r: float32[N]
            buf_i: float32[N]
            res_r: float32[N]
            res_i: float32[N]

            # Compute stage parameters from pid
            span: int32 = spans[pid]
            tw_offset: int32 = offsets[pid]
            shift: int32 = pid + 2  # stage number = pid + 2

            # Phase 1: Read all chunks from input stream
            for c in range(NUM_CHUNKS):
                tmp_r: float32[CHUNK] = s_r[pid].get()
                tmp_i: float32[CHUNK] = s_i[pid].get()
                for u in range(CHUNK):
                    idx: int32 = c * CHUNK + u
                    buf_r[idx] = tmp_r[u]
                    buf_i[idx] = tmp_i[u]

            # Phase 2: Compute butterflies
            for b in range(HALF_N):
                group: int32 = b >> shift
                pos: int32 = b & (span - 1)
                u_idx: int32 = group * (span * 2) + pos
                l_idx: int32 = u_idx + span
                wr: float32 = all_twiddles_r[tw_offset + pos]
                wi: float32 = all_twiddles_i[tw_offset + pos]
                bwr: float32 = buf_r[l_idx] * wr - buf_i[l_idx] * wi
                bwi: float32 = buf_r[l_idx] * wi + buf_i[l_idx] * wr
                res_r[u_idx] = buf_r[u_idx] + bwr
                res_i[u_idx] = buf_i[u_idx] + bwi
                res_r[l_idx] = buf_r[u_idx] - bwr
                res_i[l_idx] = buf_i[u_idx] - bwi

            # Phase 3: Write all chunks to output stream
            for c in range(NUM_CHUNKS):
                out_r: float32[CHUNK]
                out_i: float32[CHUNK]
                for u in range(CHUNK):
                    idx: int32 = c * CHUNK + u
                    out_r[u] = res_r[idx]
                    out_i[u] = res_i[idx]
                s_r[pid + 1].put(out_r)
                s_i[pid + 1].put(out_i)

        @df.kernel(mapping=[1], args=[out_real, out_imag])
        def output_store(local_out_real: float32[N], local_out_imag: float32[N]):
            for c in range(NUM_CHUNKS):
                tmp_r: float32[CHUNK] = s_r[6].get()
                tmp_i: float32[CHUNK] = s_i[6].get()
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
    parser = argparse.ArgumentParser(description="HP-FFT (Merged Stages)")
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
        project=f"fft_{N}_uf{UF}_merged.prj",
        configs={"num_output_args": 2, "frequency": 250},
        wrap_io=False,
    )
    test_fft(mod, N, UF)

    del os.environ["OMP_NUM_THREADS"]


if __name__ == "__main__":
    main()
