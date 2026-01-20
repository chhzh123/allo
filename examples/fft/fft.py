# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# HP-FFT: High-Performance FFT Implementation using Allo Dataflow
# Based on the HP-FFT paper architecture using Cooley-Tukey radix-2 DIT algorithm

import os
from math import log2, cos, sin, pi

import allo
from allo.ir.types import float32, int32, Stream, ConstExpr
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np


def bit_reverse(x, bits):
    """Reverse the bits of x for bit-reversal permutation.
    Called at Python level during meta expansion."""
    result = 0
    for i in range(bits):
        if x & (1 << i):
            result |= 1 << (bits - 1 - i)
    return result


def get_upper_idx(stage, butterfly):
    """Get upper index for butterfly at given stage."""
    span = 1 << stage
    group_size = span << 1
    group = butterfly // span
    pos_in_group = butterfly % span
    return group * group_size + pos_in_group


def get_lower_idx(stage, butterfly):
    """Get lower index for butterfly at given stage."""
    return get_upper_idx(stage, butterfly) + (1 << stage)


def get_tw_real(stage, butterfly, N):
    """Compute real part of twiddle factor."""
    span = 1 << stage
    group_size = span << 1
    pos_in_group = butterfly % span
    tw_idx = pos_in_group * (N // group_size)
    angle = -2.0 * pi * tw_idx / N
    return cos(angle)


def get_tw_imag(stage, butterfly, N):
    """Compute imaginary part of twiddle factor."""
    span = 1 << stage
    group_size = span << 1
    pos_in_group = butterfly % span
    tw_idx = pos_in_group * (N // group_size)
    angle = -2.0 * pi * tw_idx / N
    return sin(angle)


def get_fft_top(N):
    """
    Generate an N-point FFT dataflow region.

    N must be a power of 2.

    Architecture:
    - log2(N) stages of butterfly computations
    - Each stage has N/2 butterfly units
    - Streams connect stages for pipelined execution
    - Bit-reversal permutation at input
    """
    assert N > 0 and (N & (N - 1)) == 0, "N must be a power of 2"

    LOG2_N = int(log2(N))
    HALF_N = N // 2

    @df.region()
    def top(
        inp_real: float32[N],
        inp_imag: float32[N],
        out_real: float32[N],
        out_imag: float32[N],
    ):
        # Streams between stages [stage, point_index]
        stage_real: Stream[float32, 4][LOG2_N + 1, N]
        stage_imag: Stream[float32, 4][LOG2_N + 1, N]

        @df.kernel(mapping=[N], args=[inp_real, inp_imag])
        def input_loader(local_inp_real: float32[N], local_inp_imag: float32[N]):
            """Load input with bit-reversal permutation."""
            idx = df.get_pid()
            val_real: float32 = local_inp_real[idx]
            val_imag: float32 = local_inp_imag[idx]
            # bit_reverse is called with compile-time constant idx
            stage_real[0, bit_reverse(idx, LOG2_N)].put(val_real)
            stage_imag[0, bit_reverse(idx, LOG2_N)].put(val_imag)

        @df.kernel(mapping=[LOG2_N, HALF_N])
        def butterfly():
            """Butterfly computation kernel."""
            s, b = df.get_pid()

            # s and b are compile-time constants from get_pid()
            upper: ConstExpr[int32] = get_upper_idx(s, b)
            lower: ConstExpr[int32] = get_lower_idx(s, b)
            tw_r: ConstExpr[float32] = get_tw_real(s, b, N)
            tw_i: ConstExpr[float32] = get_tw_imag(s, b, N)

            # Read from current stage
            a_real: float32 = stage_real[s, upper].get()
            a_imag: float32 = stage_imag[s, upper].get()
            b_real: float32 = stage_real[s, lower].get()
            b_imag: float32 = stage_imag[s, lower].get()

            # Complex multiply: bw = b * twiddle
            bw_real: float32 = b_real * tw_r - b_imag * tw_i
            bw_imag: float32 = b_real * tw_i + b_imag * tw_r

            # Butterfly: write to next stage
            stage_real[s + 1, upper].put(a_real + bw_real)
            stage_imag[s + 1, upper].put(a_imag + bw_imag)
            stage_real[s + 1, lower].put(a_real - bw_real)
            stage_imag[s + 1, lower].put(a_imag - bw_imag)

        @df.kernel(mapping=[N], args=[out_real, out_imag])
        def output_store(local_out_real: float32[N], local_out_imag: float32[N]):
            """Store output from the final stage."""
            idx = df.get_pid()
            local_out_real[idx] = stage_real[LOG2_N, idx].get()
            local_out_imag[idx] = stage_imag[LOG2_N, idx].get()

    return top


def test_fft(N=8):
    """Test the FFT implementation against NumPy reference."""
    LOG2_N = int(log2(N))

    # Generate test input
    np.random.seed(42)
    inp_real = np.random.rand(N).astype(np.float32)
    inp_imag = np.zeros(N, dtype=np.float32)

    # Output arrays
    out_real = np.zeros(N, dtype=np.float32)
    out_imag = np.zeros(N, dtype=np.float32)

    print(f"Building {N}-point FFT ({LOG2_N} stages)...")

    # Build and run the FFT
    top = get_fft_top(N)
    sim_mod = df.build(top, target="simulator")

    print("Running simulator...")
    sim_mod(inp_real, inp_imag, out_real, out_imag)

    # NumPy reference
    ref = np.fft.fft(inp_real + 1j * inp_imag)
    ref_real = ref.real.astype(np.float32)
    ref_imag = ref.imag.astype(np.float32)

    # Verify results
    print("=" * 60)
    print("HP-FFT Test Results")
    print("=" * 60)
    print(f"FFT Size: {N} points, {LOG2_N} stages")
    print("-" * 60)

    if N <= 16:
        print("\nInput (real):", inp_real)
        print("\nOutput (real):", out_real)
        print("Reference (real):", ref_real)
        print("\nOutput (imag):", out_imag)
        print("Reference (imag):", ref_imag)
    else:
        print(f"\nInput (first 8 of {N}):", inp_real[:8])
        print(f"\nOutput (first 8 of {N}):", out_real[:8])
        print(f"Reference (first 8 of {N}):", ref_real[:8])

    max_diff_real = np.max(np.abs(out_real - ref_real))
    max_diff_imag = np.max(np.abs(out_imag - ref_imag))

    print("-" * 60)
    print(f"Max difference (real): {max_diff_real:.6e}")
    print(f"Max difference (imag): {max_diff_imag:.6e}")

    try:
        np.testing.assert_allclose(out_real, ref_real, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out_imag, ref_imag, rtol=1e-4, atol=1e-4)
        print(f"\n✅ HP-FFT {N}-point Simulator Test PASSED!")
    except AssertionError as e:
        print(f"\n❌ HP-FFT {N}-point Simulator Test FAILED: {e}")

    print("=" * 60)


if __name__ == "__main__":
    import sys

    # Default to 8-point FFT, or use command-line argument
    N = 8
    if len(sys.argv) > 1:
        N = int(sys.argv[1])

    # Set number of threads based on FFT size
    num_threads = max(64, N * 2)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    test_fft(N)

    del os.environ["OMP_NUM_THREADS"]
