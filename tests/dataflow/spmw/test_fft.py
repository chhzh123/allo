# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPMW spatial FFT: a radix-2 butterfly work unit over a key-form ``lane`` permutation topology.

An N-point Hardware-Pipelined FFT written once as an SPMW design: a single ``@spmw.unit`` butterfly
mapped over a (log2 N, N/2) grid, wired by key-form ``("lane_*", stage, slot)`` links (a permutation
network, not a mesh), and fed/drained by ``spmw.stream_in``/``spmw.stream_out`` at stages 0 and S.
Every butterfly reads two lanes at its stage and writes two lanes at the next -- the non-affine
``{upper, lower}`` key access pattern. The spatial variant is fully unrolled (one PE per butterfly,
all FIFOs); building it on the simulator and matching ``numpy.fft.fft`` proves the key-form topology
path lowers correctly. The pure-Python helpers are the numerical oracle shared with the design.
"""

from math import log2, cos, sin, pi

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import float32, int32, ConstExpr


def bit_reverse(x, bits):
    """Reverse the low ``bits`` bits of x (the stage-0 input permutation)."""
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


def _fft_region(N):
    """An N-point spatial FFT as an SPMW region (butterfly unit + key-form ``lane`` topology)."""
    assert N > 0 and (N & (N - 1)) == 0, "N must be a power of 2"
    S = int(log2(N))
    HALF = N // 2

    @spmw.unit
    def bfly(ctx):
        s, b = ctx.rank()
        a_re: float32 = ctx.a_re.get()
        a_im: float32 = ctx.a_im.get()
        b_re: float32 = ctx.b_re.get()
        b_im: float32 = ctx.b_im.get()
        tw_r: ConstExpr[float32] = get_tw_real(s, b, N)
        tw_i: ConstExpr[float32] = get_tw_imag(s, b, N)
        bw_re: float32 = b_re * tw_r - b_im * tw_i
        bw_im: float32 = b_re * tw_i + b_im * tw_r
        ctx.y_re.put(a_re + bw_re)
        ctx.y_im.put(a_im + bw_im)
        ctx.z_re.put(a_re - bw_re)
        ctx.z_im.put(a_im - bw_im)

    def bfly_links(s, b):
        up = get_upper_idx(s, b)
        lo = get_lower_idx(s, b)
        return {
            "a_re": (("lane_re", s, up), "sink"),
            "a_im": (("lane_im", s, up), "sink"),
            "b_re": (("lane_re", s, lo), "sink"),
            "b_im": (("lane_im", s, lo), "sink"),
            "y_re": (("lane_re", s + 1, up), "src"),
            "y_im": (("lane_im", s + 1, up), "src"),
            "z_re": (("lane_re", s + 1, lo), "src"),
            "z_im": (("lane_im", s + 1, lo), "src"),
        }

    topo = spmw.Topology(grid=(S, HALF), link=bfly_links)

    @spmw.region()
    def fft_spatial(
        Xr: float32[N],
        Xi: float32[N],
        Yr: float32[N],
        Yi: float32[N],
    ):
        spmw.map(bfly, grid=(S, HALF), topo=topo)
        spmw.stream_in(
            (Xr, Xi), into=("lane_re", "lane_im"), at_stage=0, index=bit_reverse
        )
        spmw.stream_out((Yr, Yi), from_=("lane_re", "lane_im"), at_stage=S)

    return fft_spatial


@pytest.mark.parametrize("N", [8, 16, 32])
def test_fft_spatial(N):
    """The SPMW spatial FFT matches numpy.fft.fft on the simulator (fft_spatial sim-correct)."""
    np.random.seed(42)
    inp_real = np.random.rand(N).astype(np.float32)
    inp_imag = np.zeros(N, dtype=np.float32)
    out_real = np.zeros(N, dtype=np.float32)
    out_imag = np.zeros(N, dtype=np.float32)
    sim_mod = spmw.build(_fft_region(N), target="simulator")
    sim_mod(inp_real, inp_imag, out_real, out_imag)
    ref = np.fft.fft(inp_real + 1j * inp_imag)
    np.testing.assert_allclose(
        out_real, ref.real.astype(np.float32), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        out_imag, ref.imag.astype(np.float32), rtol=1e-4, atol=1e-4
    )
