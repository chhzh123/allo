# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HP-FFT: High-Performance Streaming FFT (Generalized Version)

256-point FFT with configurable unroll factor (UF=1, 2, 4).
Uses stream arrays and dynamic indexing to avoid code duplication.

Architecture:
- Intra-chunk stages: butterflies where span < CHUNK (true streaming)
- Cross-chunk stages: butterflies where span >= CHUNK (3-phase pattern)

The number of intra-chunk vs cross-chunk stages depends on UF:
- UF=1: CHUNK=2, 1 intra-chunk stage (stage 0), 7 cross-chunk stages (1-7)
- UF=2: CHUNK=4, 2 intra-chunk stages (0-1), 6 cross-chunk stages (2-7)
- UF=4: CHUNK=8, 3 intra-chunk stages (0-2), 5 cross-chunk stages (3-7)
"""

import os
import argparse
from math import log2, cos, sin, pi

import allo
from allo.ir.types import float32, int32, Stream
import allo.dataflow as df
import numpy as np


def bit_reverse(x, bits):
    result = 0
    for i in range(bits):
        if x & (1 << i):
            result |= 1 << (bits - 1 - i)
    return result


def create_fft(N: int, UF: int):
    """Create FFT dataflow for N-point FFT with given unroll factor."""
    if UF not in [1, 2, 4]:
        raise ValueError(f"UF must be 1, 2, or 4. Got UF={UF}")

    LOG2_N = int(log2(N))
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
    # Maximum 3 stages for UF=4: stages 0, 1, 2
    intra_tw_r = np.concatenate([make_twiddles(s)[0] for s in range(LOG2_CHUNK)])
    intra_tw_i = np.concatenate([make_twiddles(s)[1] for s in range(LOG2_CHUNK)])
    # Offsets for intra-chunk twiddles: 0, 1, 3 (for stages 0, 1, 2)
    np_intra_offsets = np.array(
        [sum(1 << s for s in range(k)) for k in range(LOG2_CHUNK)], dtype=np.int32
    )
    np_intra_spans = np.array([1 << s for s in range(LOG2_CHUNK)], dtype=np.int32)

    # Concatenate twiddles for all cross-chunk stages (LOG2_CHUNK to LOG2_N-1)
    cross_tw_r = np.concatenate(
        [make_twiddles(s)[0] for s in range(LOG2_CHUNK, LOG2_N)]
    )
    cross_tw_i = np.concatenate(
        [make_twiddles(s)[1] for s in range(LOG2_CHUNK, LOG2_N)]
    )
    # Offsets for cross-chunk twiddles
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

    @df.region()
    def top(
        inp_real: float32[N],
        inp_imag: float32[N],
        out_real: float32[N],
        out_imag: float32[N],
    ):
        # Stream arrays for intra-chunk stages (indices 0 to NUM_INTRA)
        # intra_s[0] = input from loader
        # intra_s[NUM_INTRA] = output to cross-chunk stages
        intra_s_r: Stream[float32[CHUNK], 4][NUM_INTRA + 1]
        intra_s_i: Stream[float32[CHUNK], 4][NUM_INTRA + 1]

        # Stream arrays for cross-chunk stages (indices 0 to NUM_CROSS)
        # cross_s[0] = input from intra-chunk stages
        # cross_s[NUM_CROSS] = output to output_store
        cross_s_r: Stream[float32[CHUNK], 4][NUM_CROSS + 1]
        cross_s_i: Stream[float32[CHUNK], 4][NUM_CROSS + 1]

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
                intra_s_r[0].put(buf_r)
                intra_s_i[0].put(buf_i)

        # =================================================================
        # INTRA-CHUNK STAGES: True streaming, butterflies within chunk
        #
        # For stage s: span = 2^s, butterflies pair elements at distance span
        # =================================================================
        @df.kernel(mapping=[NUM_INTRA])
        def intra_chunk_stage():
            pid = df.get_pid()
            all_twiddles_r: float32[CHUNK - 1] = intra_tw_r
            all_twiddles_i: float32[CHUNK - 1] = intra_tw_i
            offsets: int32[NUM_INTRA] = np_intra_offsets
            spans: int32[NUM_INTRA] = np_intra_spans

            span: int32 = spans[pid]
            tw_offset: int32 = offsets[pid]

            for c in range(NUM_CHUNKS):
                ir: float32[CHUNK] = intra_s_r[pid].get()
                ii: float32[CHUNK] = intra_s_i[pid].get()
                or_out: float32[CHUNK]
                oi_out: float32[CHUNK]

                # Number of butterfly groups in this stage
                # For stage 0: groups = CHUNK/2, each group has 1 butterfly
                # For stage 1: groups = CHUNK/4, each group has 2 butterflies
                # Generally: num_groups = CHUNK / (2 * span)
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
        #
        # Each stage reads all chunks, computes butterflies, writes all chunks
        # =================================================================
        @df.kernel(mapping=[NUM_CROSS])
        def cross_chunk_stage():
            pid = df.get_pid()
            all_twiddles_r: float32[N - CHUNK] = cross_tw_r
            all_twiddles_i: float32[N - CHUNK] = cross_tw_i
            offsets: int32[NUM_CROSS] = np_cross_offsets
            spans: int32[NUM_CROSS] = np_cross_spans

            buf_r: float32[N]
            buf_i: float32[N]
            res_r: float32[N]
            res_i: float32[N]

            # Compute stage parameters from pid
            span: int32 = spans[pid]
            tw_offset: int32 = offsets[pid]
            shift: int32 = pid + LOG2_CHUNK  # stage number

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
            for c in range(NUM_CHUNKS):
                tmp_r: float32[CHUNK] = cross_s_r[NUM_CROSS].get()
                tmp_i: float32[CHUNK] = cross_s_i[NUM_CROSS].get()
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
    print(f"HP-FFT {N}-point ({LOG2_N} stages, UF={UF})")
    print(
        f"  CHUNK={UF*2}, {LOG2_CHUNK} intra-chunk stages, {LOG2_N - LOG2_CHUNK} cross-chunk stages"
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
    parser.add_argument("--uf", type=int, default=2, choices=[1, 2, 4])
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
