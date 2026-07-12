# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPMW ping-pong GEMM: an operand pinned with ``spmw.shared(double=True)`` (M4, plan.md §6.7).

``spmw.shared(..., double=True)`` requests ping-pong double buffering: two alternating physical
copies of the buffer (the ``buffer_at`` path) so the next tile can be filled while the current one is
consumed. It is a throughput/scheduling choice -- functionally transparent -- so the simulator result
is ``A @ B`` regardless; the flag rides the rolled ``spmw.map`` IR (``spmw.double_buffers``) and the
rolled HLS emitter realizes it as a true dual-port (``RAM_T2P``) storage that supports the concurrent
read/write double-buffering needs, rather than being silently dropped.
"""

import os
import re
import subprocess
import tempfile

import numpy as np
import pytest

import allo.spmw as spmw
import allo.backend.hls as hls
from allo.spmw_hls import emit_rolled_hls_ir
from allo.ir.types import float32


def _gemm(M, N, K, double_b=False):
    """A systolic GEMM twin that optionally pins its B operand to a ping-pong double buffer."""
    grid = spmw.mesh((M, N))

    @spmw.unit
    def pe(ctx):
        c: float32 = 0
        for k in range(K):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")
        if double_b:
            spmw.place("B", spmw.shared(float32, space="L2", double=True))

    return gemm


def test_shared_double_carries_the_ping_pong_flag():
    assert spmw.shared(float32, double=True).double is True
    assert spmw.shared(float32).double is False  # default is single-buffered


def test_pingpong_placement_round_trips_on_the_rolled_ir():
    """A ``double=True`` placement rides the rolled IR on the ``spmw.double_buffers`` string list."""
    ir = str(spmw.lower(_gemm(4, 4, 4, double_b=True)))
    match = re.search(r"spmw\.double_buffers = \[([^\]]*)\]", ir)
    assert match and '"B"' in match.group(1)
    # a single-buffered map carries no such attribute
    assert "spmw.double_buffers" not in str(spmw.lower(_gemm(4, 4, 4, double_b=False)))


def test_pingpong_emitter_double_buffers_the_operand():
    """The rolled HLS emitter binds the double-buffered operand to a true dual-port RAM."""
    cpp = emit_rolled_hls_ir(_gemm(4, 4, 4, double_b=True))
    assert "#pragma HLS bind_storage variable=B type=RAM_T2P" in cpp
    # a single-buffered map does not emit the ping-pong storage
    assert "RAM_T2P" not in emit_rolled_hls_ir(_gemm(4, 4, 4, double_b=False))


def test_pingpong_gemm_simulator_matches_numpy():
    """L1: the ping-pong GEMM (double-buffered B) is simulator-correct against numpy (double buffering
    is functionally transparent, so the result is still A @ B)."""
    M, N, K = 4, 4, 8
    np.random.seed(0)
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    sim_mod = spmw.build(_gemm(M, N, K, double_b=True), target="simulator")
    sim_mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-4)


@pytest.mark.skipif(
    not hls.is_available("vitis_hls"), reason="requires the Vitis HLS toolchain"
)
def test_pingpong_gemm_rolled_csim():
    """L2: the double-buffered rolled top csims correct on Vitis HLS (the RAM_T2P bind is valid)."""
    with tempfile.TemporaryDirectory() as tmp:
        spmw.build(
            _gemm(4, 4, 4, double_b=True), target="rolled", project=tmp, testbench=True
        )
        result = subprocess.run(
            ["vitis_hls", "-f", "run.tcl"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=900,
            check=True,
        )
    assert "CSIM MATCH" in result.stdout, result.stdout[-2000:]
