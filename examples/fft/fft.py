# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HP-FFT: High-Performance Streaming FFT (Generalized Version)

256-point FFT with configurable unroll factor (any power of 2 from 1 to N/2).
Uses stream arrays and dynamic indexing to avoid code duplication.

Architecture:
- Intra-chunk stages: butterflies where span < CHUNK (true streaming)
- Cross-chunk stages: butterflies where span >= CHUNK (3-phase pattern)

The number of intra-chunk vs cross-chunk stages depends on UF:
- UF=1:   CHUNK=2,   1 intra-chunk stage,  7 cross-chunk stages
- UF=2:   CHUNK=4,   2 intra-chunk stages, 6 cross-chunk stages
- UF=4:   CHUNK=8,   3 intra-chunk stages, 5 cross-chunk stages
- UF=8:   CHUNK=16,  4 intra-chunk stages, 4 cross-chunk stages
- UF=16:  CHUNK=32,  5 intra-chunk stages, 3 cross-chunk stages
- UF=32:  CHUNK=64,  6 intra-chunk stages, 2 cross-chunk stages
- UF=64:  CHUNK=128, 7 intra-chunk stages, 1 cross-chunk stage
- UF=128: CHUNK=256, 8 intra-chunk stages, 0 cross-chunk stages (fully unrolled)
"""

import os
import argparse
from math import log2, cos, sin, pi

import allo
from allo.ir.types import float32, int32, Stream, ConstExpr
import allo.dataflow as df
import numpy as np


def bit_reverse(x, bits):
    result = 0
    for i in range(bits):
        if x & (1 << i):
            result |= 1 << (bits - 1 - i)
    return result


def create_fft(N: int, UF: int):
    """Create FFT dataflow for N-point FFT with given unroll factor.

    UF must be a power of 2 in range [1, N/2].
    """
    LOG2_N = int(log2(N))
    LOG2_UF = int(log2(UF))

    # Validate UF
    if UF < 1 or UF > N // 2:
        raise ValueError(f"UF must be in range [1, {N//2}]. Got UF={UF}")
    if (UF & (UF - 1)) != 0:
        raise ValueError(f"UF must be a power of 2. Got UF={UF}")

    HALF_N = N // 2
    CHUNK = UF * 2
    NUM_CHUNKS = N // CHUNK
    LOG2_CHUNK = int(log2(CHUNK))  # Number of intra-chunk stages
    NUM_INTRA = LOG2_CHUNK  # stages 0 to LOG2_CHUNK-1
    NUM_CROSS = LOG2_N - LOG2_CHUNK  # stages LOG2_CHUNK to LOG2_N-1

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

    # Concatenate twiddles for all intra-chunk stages (0 to LOG2_CHUNK-1)
    intra_tw_r = np.concatenate([make_twiddles(s)[0] for s in range(LOG2_CHUNK)])
    intra_tw_i = np.concatenate([make_twiddles(s)[1] for s in range(LOG2_CHUNK)])
    # Offsets for intra-chunk twiddles
    np_intra_offsets = np.array(
        [sum(1 << s for s in range(k)) for k in range(LOG2_CHUNK)], dtype=np.int32
    )
    np_intra_spans = np.array([1 << s for s in range(LOG2_CHUNK)], dtype=np.int32)

    # Concatenate twiddles for all cross-chunk stages (LOG2_CHUNK to LOG2_N-1)
    if NUM_CROSS > 0:
        cross_tw_r = np.concatenate(
            [make_twiddles(s)[0] for s in range(LOG2_CHUNK, LOG2_N)]
        )
        cross_tw_i = np.concatenate(
            [make_twiddles(s)[1] for s in range(LOG2_CHUNK, LOG2_N)]
        )
        np_cross_offsets = np.array(
            [
                sum(1 << s for s in range(LOG2_CHUNK, LOG2_CHUNK + k))
                for k in range(NUM_CROSS)
            ],
            dtype=np.int32,
        )
        np_cross_spans = np.array(
            [1 << s for s in range(LOG2_CHUNK, LOG2_N)], dtype=np.int32
        )
        CROSS_TW_SIZE = N - CHUNK
    else:
        # Fully unrolled case: no cross-chunk stages
        cross_tw_r = np.zeros(1, dtype=np.float32)
        cross_tw_i = np.zeros(1, dtype=np.float32)
        np_cross_offsets = np.zeros(1, dtype=np.int32)
        np_cross_spans = np.zeros(1, dtype=np.int32)
        CROSS_TW_SIZE = 1

    INTRA_TW_SIZE = CHUNK - 1

    @df.region()
    def top(
        inp_real: float32[N],
        inp_imag: float32[N],
        out_real: float32[N],
        out_imag: float32[N],
    ):
        # Stream arrays for intra-chunk stages (indices 0 to NUM_INTRA)
        intra_s_r: Stream[float32[CHUNK], 4][NUM_INTRA + 1]
        intra_s_i: Stream[float32[CHUNK], 4][NUM_INTRA + 1]

        # Stream arrays for cross-chunk stages (indices 0 to NUM_CROSS)
        # If NUM_CROSS == 0, we still need 1 stream to connect intra to output
        cross_s_r: Stream[float32[CHUNK], 4][max(NUM_CROSS, 1) + 1]
        cross_s_i: Stream[float32[CHUNK], 4][max(NUM_CROSS, 1) + 1]

        # Input loader with bit-reversal
        @df.kernel(mapping=[1], args=[inp_real, inp_imag])
        def input_loader(local_inp_real: float32[N], local_inp_imag: float32[N]):
            bit_rev: int32[N] = np_bit_rev
            buf_in_r: float32[N]
            buf_in_i: float32[N]
            # 1. Sequential load to enable burst
            for i in range(N):
                buf_in_r[i] = local_inp_real[i]
                buf_in_i[i] = local_inp_imag[i]
            # 2. Bit-reversal shuffle from local buffer
            for c in range(NUM_CHUNKS):
                buf_r: float32[CHUNK]
                buf_i: float32[CHUNK]
                for u in range(CHUNK):
                    src: int32 = c * CHUNK + u
                    rev: int32 = bit_rev[src]
                    buf_r[u] = buf_in_r[rev]
                    buf_i[u] = buf_in_i[rev]
                intra_s_r[0].put(buf_r)
                intra_s_i[0].put(buf_i)

        # =================================================================
        # INTRA-CHUNK STAGES: True streaming, butterflies within chunk
        # =================================================================
        @df.kernel(mapping=[NUM_INTRA])
        def intra_chunk_stage():
            pid = df.get_pid()
            all_twiddles_r: float32[INTRA_TW_SIZE] = intra_tw_r
            all_twiddles_i: float32[INTRA_TW_SIZE] = intra_tw_i
            offsets: int32[NUM_INTRA] = np_intra_offsets
            spans: int32[NUM_INTRA] = np_intra_spans

            span: int32 = spans[pid]
            tw_offset: int32 = offsets[pid]

            for c in range(NUM_CHUNKS):
                ir: float32[CHUNK] = intra_s_r[pid].get()
                ii: float32[CHUNK] = intra_s_i[pid].get()
                or_out: float32[CHUNK]
                oi_out: float32[CHUNK]

                # Number of butterfly groups: CHUNK / (2 * span)
                for g in range(CHUNK // (span * 2)):
                    for k in range(span):
                        u_idx: int32 = g * (span * 2) + k
                        l_idx: int32 = u_idx + span
                        wr: float32 = all_twiddles_r[tw_offset + k]
                        wi: float32 = all_twiddles_i[tw_offset + k]
                        bwr: float32 = ir[l_idx] * wr - ii[l_idx] * wi
                        bwi: float32 = ir[l_idx] * wi + ii[l_idx] * wr
                        or_out[u_idx] = ir[u_idx] + bwr
                        oi_out[u_idx] = ii[u_idx] + bwi
                        or_out[l_idx] = ir[u_idx] - bwr
                        oi_out[l_idx] = ii[u_idx] - bwi

                intra_s_r[pid + 1].put(or_out)
                intra_s_i[pid + 1].put(oi_out)

        # Bridge: connect intra-chunk output to cross-chunk input
        @df.kernel(mapping=[1])
        def bridge():
            for c in range(NUM_CHUNKS):
                tmp_r: float32[CHUNK] = intra_s_r[NUM_INTRA].get()
                tmp_i: float32[CHUNK] = intra_s_i[NUM_INTRA].get()
                cross_s_r[0].put(tmp_r)
                cross_s_i[0].put(tmp_i)

        # =================================================================
        # CROSS-CHUNK STAGES: 3-phase pattern (read all, compute, write all)
        # Only created if NUM_CROSS > 0
        # =================================================================
        with allo.meta_if(NUM_CROSS > 0):

            @df.kernel(mapping=[NUM_CROSS])
            def cross_chunk_stage():
                pid = df.get_pid()
                all_twiddles_r: float32[CROSS_TW_SIZE] = cross_tw_r
                all_twiddles_i: float32[CROSS_TW_SIZE] = cross_tw_i
                offsets: int32[NUM_CROSS] = np_cross_offsets
                spans: int32[NUM_CROSS] = np_cross_spans

                buf_r: float32[N]
                buf_i: float32[N]
                res_r: float32[N]
                res_i: float32[N]

                span: int32 = spans[pid]
                tw_offset: int32 = offsets[pid]
                shift: int32 = pid + LOG2_CHUNK

                # Phase 1: Read all chunks from input stream
                for c in range(NUM_CHUNKS):
                    tmp_r: float32[CHUNK] = cross_s_r[pid].get()
                    tmp_i: float32[CHUNK] = cross_s_i[pid].get()
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
                    cross_s_r[pid + 1].put(out_r)
                    cross_s_i[pid + 1].put(out_i)

        @df.kernel(mapping=[1], args=[out_real, out_imag])
        def output_store(local_out_real: float32[N], local_out_imag: float32[N]):
            # Read from appropriate stream based on whether cross-chunk stages exist
            out_stream_idx: ConstExpr[int32] = NUM_CROSS if NUM_CROSS > 0 else 0
            for c in range(NUM_CHUNKS):
                tmp_r: float32[CHUNK] = cross_s_r[out_stream_idx].get()
                tmp_i: float32[CHUNK] = cross_s_i[out_stream_idx].get()
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
    LOG2_CHUNK = int(log2(UF * 2))
    NUM_INTRA = LOG2_CHUNK
    NUM_CROSS = LOG2_N - LOG2_CHUNK
    print(f"HP-FFT {N}-point ({LOG2_N} stages, UF={UF})")
    print(
        f"  CHUNK={UF*2}, {NUM_INTRA} intra-chunk stages, {NUM_CROSS} cross-chunk stages"
    )

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
    parser = argparse.ArgumentParser(description="HP-FFT (Generalized)")
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

    s = df.customize(top)
    if UF > 1:
        # Partition internal buffers in input_loader
        chunk_size = UF * 2
        for buffer in ["buf_in_r", "buf_in_i"]:
            s.partition(
                f"input_loader_0:{buffer}", dim=0, partition_type=2, factor=chunk_size
            )

    mod = s.build(
        target="vitis_hls",
        mode="hw_emu",
        project=f"fft_{N}_uf{UF}.prj",
        configs={"num_output_args": 2, "frequency": 250},
        wrap_io=False,
    )
    test_fft(mod, N, UF)


if __name__ == "__main__":
    main()
