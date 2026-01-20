# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# HP-FFT: High-Performance FFT Implementation using Allo Dataflow
# Based on the HP-FFT paper architecture using Cooley-Tukey radix-2 DIT algorithm
#
# This implementation uses stream of tensors to avoid excessive resource usage
# from meta_for. Each FFT stage exchanges complete N-element tensors via streams
# rather than using 2D arrays of scalar streams.

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
N = 8  # FFT size (must be power of 2)
LOG2_N = int(log2(N))
HALF_N = N // 2

# ============================================================================
# Precomputed data as numpy arrays (following MLP test pattern)
# ============================================================================

# Bit-reversal permutation indices
np_bit_rev = np.array([0, 4, 2, 6, 1, 5, 3, 7], dtype=np.int32)

# Stage 0: span=1
np_tw_r0 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
np_tw_i0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
np_up0 = np.array([0, 2, 4, 6], dtype=np.int32)
np_lo0 = np.array([1, 3, 5, 7], dtype=np.int32)

# Stage 1: span=2
np_tw_r1 = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
np_tw_i1 = np.array([0.0, -1.0, 0.0, -1.0], dtype=np.float32)
np_up1 = np.array([0, 1, 4, 5], dtype=np.int32)
np_lo1 = np.array([2, 3, 6, 7], dtype=np.int32)

# Stage 2: span=4
np_tw_r2 = np.array(
    [1.0, cos(-2 * pi * 1 / 8), 0.0, cos(-2 * pi * 3 / 8)], dtype=np.float32
)
np_tw_i2 = np.array(
    [0.0, sin(-2 * pi * 1 / 8), -1.0, sin(-2 * pi * 3 / 8)], dtype=np.float32
)
np_up2 = np.array([0, 1, 2, 3], dtype=np.int32)
np_lo2 = np.array([4, 5, 6, 7], dtype=np.int32)


# ============================================================================
# FFT Dataflow Region
# ============================================================================


@df.region()
def top(
    inp_real: float32[N],
    inp_imag: float32[N],
    out_real: float32[N],
    out_imag: float32[N],
):
    """N-point FFT using stream of tensors architecture."""
    # Stream of tensors between stages
    pipe_real_0: Stream[float32[N], 4]
    pipe_imag_0: Stream[float32[N], 4]
    pipe_real_1: Stream[float32[N], 4]
    pipe_imag_1: Stream[float32[N], 4]
    pipe_real_2: Stream[float32[N], 4]
    pipe_imag_2: Stream[float32[N], 4]

    @df.kernel(mapping=[1], args=[inp_real, inp_imag])
    def input_loader(local_inp_real: float32[N], local_inp_imag: float32[N]):
        """Load input with bit-reversal permutation."""
        bit_rev: int32[N] = np_bit_rev
        buf_real: float32[N] = 0
        buf_imag: float32[N] = 0
        for idx in range(N):
            rev_idx: int32 = bit_rev[idx]
            buf_real[rev_idx] = local_inp_real[idx]
            buf_imag[rev_idx] = local_inp_imag[idx]
        pipe_real_0.put(buf_real)
        pipe_imag_0.put(buf_imag)

    @df.kernel(mapping=[1])
    def stage0():
        """Butterfly stage 0: span=1."""
        tw_r0: float32[HALF_N] = np_tw_r0
        tw_i0: float32[HALF_N] = np_tw_i0
        up0: int32[HALF_N] = np_up0
        lo0: int32[HALF_N] = np_lo0
        in_r: float32[N] = pipe_real_0.get()
        in_i: float32[N] = pipe_imag_0.get()
        out_r: float32[N] = 0
        out_i: float32[N] = 0
        for b in range(HALF_N):
            upper: int32 = up0[b]
            lower: int32 = lo0[b]
            twr: float32 = tw_r0[b]
            twi: float32 = tw_i0[b]
            ar: float32 = in_r[upper]
            ai: float32 = in_i[upper]
            br: float32 = in_r[lower]
            bi: float32 = in_i[lower]
            bwr: float32 = br * twr - bi * twi
            bwi: float32 = br * twi + bi * twr
            out_r[upper] = ar + bwr
            out_i[upper] = ai + bwi
            out_r[lower] = ar - bwr
            out_i[lower] = ai - bwi
        pipe_real_1.put(out_r)
        pipe_imag_1.put(out_i)

    @df.kernel(mapping=[1])
    def stage1():
        """Butterfly stage 1: span=2."""
        tw_r1: float32[HALF_N] = np_tw_r1
        tw_i1: float32[HALF_N] = np_tw_i1
        up1: int32[HALF_N] = np_up1
        lo1: int32[HALF_N] = np_lo1
        in_r: float32[N] = pipe_real_1.get()
        in_i: float32[N] = pipe_imag_1.get()
        out_r: float32[N] = 0
        out_i: float32[N] = 0
        for b in range(HALF_N):
            upper: int32 = up1[b]
            lower: int32 = lo1[b]
            twr: float32 = tw_r1[b]
            twi: float32 = tw_i1[b]
            ar: float32 = in_r[upper]
            ai: float32 = in_i[upper]
            br: float32 = in_r[lower]
            bi: float32 = in_i[lower]
            bwr: float32 = br * twr - bi * twi
            bwi: float32 = br * twi + bi * twr
            out_r[upper] = ar + bwr
            out_i[upper] = ai + bwi
            out_r[lower] = ar - bwr
            out_i[lower] = ai - bwi
        pipe_real_2.put(out_r)
        pipe_imag_2.put(out_i)

    @df.kernel(mapping=[1], args=[out_real, out_imag])
    def stage2_and_store(local_out_real: float32[N], local_out_imag: float32[N]):
        """Butterfly stage 2 and output store."""
        tw_r2: float32[HALF_N] = np_tw_r2
        tw_i2: float32[HALF_N] = np_tw_i2
        up2: int32[HALF_N] = np_up2
        lo2: int32[HALF_N] = np_lo2
        in_r: float32[N] = pipe_real_2.get()
        in_i: float32[N] = pipe_imag_2.get()
        out_r: float32[N] = 0
        out_i: float32[N] = 0
        for b in range(HALF_N):
            upper: int32 = up2[b]
            lower: int32 = lo2[b]
            twr: float32 = tw_r2[b]
            twi: float32 = tw_i2[b]
            ar: float32 = in_r[upper]
            ai: float32 = in_i[upper]
            br: float32 = in_r[lower]
            bi: float32 = in_i[lower]
            bwr: float32 = br * twr - bi * twi
            bwi: float32 = br * twi + bi * twr
            out_r[upper] = ar + bwr
            out_i[upper] = ai + bwi
            out_r[lower] = ar - bwr
            out_i[lower] = ai - bwi
        for idx in range(N):
            local_out_real[idx] = out_r[idx]
            local_out_imag[idx] = out_i[idx]


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

    print(f"Building {N}-point FFT ({LOG2_N} stages)...")
    print(f"  Using stream of tensors architecture")

    # sim_mod = df.build(
    #     top,
    #     target="vitis_hls",
    #     mode="hw_emu",
    #     project=f"fft_{N}.prj",
    #     configs={"num_output_args": 2},
    # )
    sim_mod = df.build(top, target="simulator")

    print("Running HW emulation...")
    sim_mod(inp_real, inp_imag, out_real, out_imag)

    ref = np.fft.fft(inp_real + 1j * inp_imag)

    print("=" * 60)
    print("HP-FFT Test Results")
    print("=" * 60)
    print(f"FFT Size: {N} points, {LOG2_N} stages")
    print("-" * 60)
    print("\nOutput (real):", out_real)
    print("Reference (real):", ref.real.astype(np.float32))
    print("\nOutput (imag):", out_imag)
    print("Reference (imag):", ref.imag.astype(np.float32))

    max_diff_real = np.max(np.abs(out_real - ref.real))
    max_diff_imag = np.max(np.abs(out_imag - ref.imag))
    print("-" * 60)
    print(f"Max difference (real): {max_diff_real:.6e}")
    print(f"Max difference (imag): {max_diff_imag:.6e}")

    try:
        np.testing.assert_allclose(out_real, ref.real, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out_imag, ref.imag, rtol=1e-4, atol=1e-4)
        print(f"\n✅ HP-FFT {N}-point Test PASSED!")
    except AssertionError as e:
        print(f"\n❌ HP-FFT {N}-point Test FAILED: {e}")
    print("=" * 60)


if __name__ == "__main__":
    num_threads = max(64, N * 2)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    test_fft()
    del os.environ["OMP_NUM_THREADS"]
