# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scalar HP-FFT (fft_spatial), ported verbatim from the ``feature/allo-fft`` branch (task4.4).

An N-point Hardware-Pipelined FFT as an ``allo.dataflow`` region: an input loader (N PEs, bit-reversal
addressing), a butterfly stage (LOG2_N x N/2 PEs, each computing one radix-2 butterfly with a
compile-time (``ConstExpr``) twiddle and upper/lower indices -- the non-affine key access pattern), and
an output store. This is the *spatial* FFT: every butterfly is its own PE. It builds on the simulator
and matches ``numpy.fft.fft`` (the ``fft_spatial`` csim-correct half of the M4 FFT gate). The folded
FFT + its conflict-free F2 banking is the next milestone.
"""

from math import log2, cos, sin, pi

import numpy as np
import pytest

import allo.dataflow as df
from allo.ir.types import float32, int32, ConstExpr, Stream


def bit_reverse(x, bits):
    """Reverse the low ``bits`` bits of x; evaluated at kernel-expansion (compile) time."""
    result = 0
    for i in range(bits):
        if x & (1 << i):
            result |= 1 << (bits - 1 - i)
    return result


def get_upper_idx(stage, butterfly):
    span = 1 << stage
    group_size = span << 1
    group = butterfly // span
    pos_in_group = butterfly % span
    return group * group_size + pos_in_group


def get_lower_idx(stage, butterfly):
    return get_upper_idx(stage, butterfly) + (1 << stage)


def get_tw_real(stage, butterfly, n_points):
    span = 1 << stage
    group_size = span << 1
    pos_in_group = butterfly % span
    tw_idx = pos_in_group * (n_points // group_size)
    return cos(-2.0 * pi * tw_idx / n_points)


def get_tw_imag(stage, butterfly, n_points):
    span = 1 << stage
    group_size = span << 1
    pos_in_group = butterfly % span
    tw_idx = pos_in_group * (n_points // group_size)
    return sin(-2.0 * pi * tw_idx / n_points)


def get_fft_top(N_):
    """An N-point scalar HP-FFT dataflow region (input_loader -> butterfly stages -> output_store)."""
    assert N_ > 0 and (N_ & (N_ - 1)) == 0, "N must be a power of 2"
    LOG2_N_ = int(log2(N_))
    HALF_N = N_ // 2

    @df.region()
    def top(
        inp_real: float32[N_],
        inp_imag: float32[N_],
        out_real: float32[N_],
        out_imag: float32[N_],
    ):
        stage_real: Stream[float32, 4][LOG2_N_ + 1, N_]
        stage_imag: Stream[float32, 4][LOG2_N_ + 1, N_]

        @df.kernel(mapping=[N_], args=[inp_real, inp_imag])
        def input_loader(local_inp_real: float32[N_], local_inp_imag: float32[N_]):
            idx = df.get_pid()
            val_real: float32 = local_inp_real[idx]
            val_imag: float32 = local_inp_imag[idx]
            stage_real[0, bit_reverse(idx, LOG2_N_)].put(val_real)
            stage_imag[0, bit_reverse(idx, LOG2_N_)].put(val_imag)

        @df.kernel(mapping=[LOG2_N_, HALF_N])
        def butterfly():
            s, b = df.get_pid()
            upper: ConstExpr[int32] = get_upper_idx(s, b)
            lower: ConstExpr[int32] = get_lower_idx(s, b)
            tw_r: ConstExpr[float32] = get_tw_real(s, b, N_)
            tw_i: ConstExpr[float32] = get_tw_imag(s, b, N_)
            a_real: float32 = stage_real[s, upper].get()
            a_imag: float32 = stage_imag[s, upper].get()
            b_real: float32 = stage_real[s, lower].get()
            b_imag: float32 = stage_imag[s, lower].get()
            bw_real: float32 = b_real * tw_r - b_imag * tw_i
            bw_imag: float32 = b_real * tw_i + b_imag * tw_r
            stage_real[s + 1, upper].put(a_real + bw_real)
            stage_imag[s + 1, upper].put(a_imag + bw_imag)
            stage_real[s + 1, lower].put(a_real - bw_real)
            stage_imag[s + 1, lower].put(a_imag - bw_imag)

        @df.kernel(mapping=[N_], args=[out_real, out_imag])
        def output_store(local_out_real: float32[N_], local_out_imag: float32[N_]):
            idx = df.get_pid()
            local_out_real[idx] = stage_real[LOG2_N_, idx].get()
            local_out_imag[idx] = stage_imag[LOG2_N_, idx].get()

    return top


@pytest.mark.parametrize("N_", [8, 16, 32])
def test_fft_spatial(N_):
    """The scalar HP-FFT matches numpy.fft.fft on the simulator (fft_spatial csim-correct)."""
    np.random.seed(42)
    inp_real = np.random.rand(N_).astype(np.float32)
    inp_imag = np.zeros(N_, dtype=np.float32)
    out_real = np.zeros(N_, dtype=np.float32)
    out_imag = np.zeros(N_, dtype=np.float32)
    sim_mod = df.build(get_fft_top(N_), target="simulator")
    sim_mod(inp_real, inp_imag, out_real, out_imag)
    ref = np.fft.fft(inp_real + 1j * inp_imag)
    np.testing.assert_allclose(
        out_real, ref.real.astype(np.float32), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        out_imag, ref.imag.astype(np.float32), rtol=1e-4, atol=1e-4
    )
