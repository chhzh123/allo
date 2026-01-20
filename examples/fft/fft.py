# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# HP-FFT: High-Performance FFT Implementation using Allo Dataflow
# Based on the HP-FFT paper architecture using Cooley-Tukey radix-2 DIT algorithm
#
# Uses stream of tensors with configurable unrolling factor UF.
# - UF controls parallelism: UF*2 elements processed per stream transfer
# - Higher UF = more parallelism, more resources
# - Lower UF = less parallelism, fewer resources
#
# NOTE: Requires compiler support for arrays of tensor streams and get_pid() indexing

import os
from math import log2, cos, sin, pi

import allo
from allo.ir.types import float32, int32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

# ============================================================================
# FFT Configuration
# ============================================================================
N = 256  # FFT size (must be power of 2)
LOG2_N = int(log2(N))
HALF_N = N // 2

UF = 2  # Unrolling factor: process UF butterflies in parallel
# UF must be power of 2, 1 <= UF <= N/2
# UF=1: minimum resources, sequential processing
# UF=N/2: maximum resources, fully parallel

CHUNK = UF * 2  # Elements per stream transfer
NUM_CHUNKS = N // CHUNK  # Sequential iterations per stage

# ============================================================================
# Precomputed data
# ============================================================================


def bit_reverse(x, bits):
    result = 0
    for i in range(bits):
        if x & (1 << i):
            result |= 1 << (bits - 1 - i)
    return result


# Bit-reversal permutation indices
np_bit_rev = np.array([bit_reverse(i, LOG2_N) for i in range(N)], dtype=np.int32)

# Precomputed butterfly indices and twiddles for all stages
# Shape: [LOG2_N, HALF_N]
np_upper = np.zeros((LOG2_N, HALF_N), dtype=np.int32)
np_lower = np.zeros((LOG2_N, HALF_N), dtype=np.int32)
np_tw_real = np.zeros((LOG2_N, HALF_N), dtype=np.float32)
np_tw_imag = np.zeros((LOG2_N, HALF_N), dtype=np.float32)

for stage in range(LOG2_N):
    span = 1 << stage
    group_size = span << 1
    for b in range(HALF_N):
        group = b // span
        pos = b % span
        np_upper[stage, b] = group * group_size + pos
        np_lower[stage, b] = np_upper[stage, b] + span
        tw_idx = pos * (N // group_size)
        angle = -2.0 * pi * tw_idx / N
        np_tw_real[stage, b] = cos(angle)
        np_tw_imag[stage, b] = sin(angle)


# ============================================================================
# FFT Dataflow Region with Partial Unrolling
# ============================================================================


@df.region()
def top(
    inp_real: float32[N],
    inp_imag: float32[N],
    out_real: float32[N],
    out_imag: float32[N],
):
    """
    N-point FFT with partial unrolling (UF butterflies in parallel).

    Data flow:
    - Stream carries CHUNK=UF*2 elements per transfer
    - Each stage processes NUM_CHUNKS=N/CHUNK transfers sequentially
    - Within each transfer, UF butterflies computed in parallel
    """
    # Array of CHUNK-element tensor streams
    stage_real: Stream[float32[CHUNK], 4][LOG2_N + 1]
    stage_imag: Stream[float32[CHUNK], 4][LOG2_N + 1]

    @df.kernel(mapping=[1], args=[inp_real, inp_imag])
    def input_loader(local_inp_real: float32[N], local_inp_imag: float32[N]):
        """Load input with bit-reversal, stream out in CHUNK-sized blocks."""
        bit_rev: int32[N] = np_bit_rev

        # Stream NUM_CHUNKS blocks of CHUNK elements
        for chunk_id in range(NUM_CHUNKS):
            buf_r: float32[CHUNK] = 0
            buf_i: float32[CHUNK] = 0
            for u in range(CHUNK):
                src_idx: int32 = chunk_id * CHUNK + u
                rev_idx: int32 = bit_rev[src_idx]
                buf_r[u] = local_inp_real[rev_idx]
                buf_i[u] = local_inp_imag[rev_idx]
            stage_real[0].put(buf_r)
            stage_imag[0].put(buf_i)

    @df.kernel(mapping=[LOG2_N])
    def butterfly_stage():
        """
        Butterfly computation with partial unrolling.
        - Outer loop: NUM_CHUNKS sequential iterations
        - Inner loop: UF parallel butterflies (unrolled)
        """
        s = df.get_pid()

        # Load precomputed data for this stage
        tw_r: float32[HALF_N] = np_tw_real[s]
        tw_i: float32[HALF_N] = np_tw_imag[s]
        upper: int32[HALF_N] = np_upper[s]
        lower: int32[HALF_N] = np_lower[s]

        # Process NUM_CHUNKS transfers
        for chunk_id in range(NUM_CHUNKS):
            # Get CHUNK elements from stream
            in_r: float32[CHUNK] = stage_real[s].get()
            in_i: float32[CHUNK] = stage_imag[s].get()
            out_r: float32[CHUNK] = 0
            out_i: float32[CHUNK] = 0

            # UF butterflies in parallel (this loop will be unrolled)
            for u in range(UF):
                b: int32 = chunk_id * UF + u  # Global butterfly index
                u_idx: int32 = upper[b] % CHUNK  # Local index within chunk
                l_idx: int32 = lower[b] % CHUNK

                # Complex multiply: bw = in[l] * twiddle
                bwr: float32 = in_r[l_idx] * tw_r[b] - in_i[l_idx] * tw_i[b]
                bwi: float32 = in_r[l_idx] * tw_i[b] + in_i[l_idx] * tw_r[b]

                # Butterfly output
                out_r[u_idx] = in_r[u_idx] + bwr
                out_i[u_idx] = in_i[u_idx] + bwi
                out_r[l_idx] = in_r[u_idx] - bwr
                out_i[l_idx] = in_i[u_idx] - bwi

            # Put CHUNK elements to next stage
            stage_real[s + 1].put(out_r)
            stage_imag[s + 1].put(out_i)

    @df.kernel(mapping=[1], args=[out_real, out_imag])
    def output_store(local_out_real: float32[N], local_out_imag: float32[N]):
        """Collect chunks and write to output."""
        for chunk_id in range(NUM_CHUNKS):
            result_r: float32[CHUNK] = stage_real[LOG2_N].get()
            result_i: float32[CHUNK] = stage_imag[LOG2_N].get()
            for u in range(CHUNK):
                idx: int32 = chunk_id * CHUNK + u
                local_out_real[idx] = result_r[u]
                local_out_imag[idx] = result_i[u]


# ============================================================================
# Test Function
# ============================================================================


def test_fft():
    """Test the FFT implementation against NumPy reference."""
    np.random.seed(42)
    inp_real = np.random.rand(N).astype(np.float32)
    inp_imag = np.zeros(N, dtype=np.float32)
    out_real = np.zeros(N, dtype=np.float32)
    out_imag = np.zeros(N, dtype=np.float32)

    print(f"Building {N}-point FFT ({LOG2_N} stages, UF={UF})...")
    print(f"  CHUNK={CHUNK} elements per transfer")
    print(f"  NUM_CHUNKS={NUM_CHUNKS} sequential iterations per stage")

    sim_mod = df.build(top, target="simulator")

    print("Running simulator...")
    sim_mod(inp_real, inp_imag, out_real, out_imag)

    ref = np.fft.fft(inp_real + 1j * inp_imag)

    print("=" * 60)
    print("HP-FFT Test Results")
    print("=" * 60)
    print(f"FFT Size: {N} points, {LOG2_N} stages, UF={UF}")
    print("-" * 60)
    print(f"\nOutput (first 8): {out_real[:8]}")
    print(f"Reference (first 8): {ref.real[:8].astype(np.float32)}")

    max_diff_real = np.max(np.abs(out_real - ref.real))
    max_diff_imag = np.max(np.abs(out_imag - ref.imag))
    print("-" * 60)
    print(f"Max difference (real): {max_diff_real:.6e}")
    print(f"Max difference (imag): {max_diff_imag:.6e}")

    try:
        np.testing.assert_allclose(out_real, ref.real, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out_imag, ref.imag, rtol=1e-4, atol=1e-4)
        print(f"\n✅ HP-FFT {N}-point (UF={UF}) Test PASSED!")
    except AssertionError as e:
        print(f"\n❌ HP-FFT {N}-point (UF={UF}) Test FAILED: {e}")
    print("=" * 60)


if __name__ == "__main__":
    num_threads = max(64, N * 2)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    test_fft()
    del os.environ["OMP_NUM_THREADS"]
