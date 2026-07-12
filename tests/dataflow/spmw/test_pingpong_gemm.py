# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPMW ping-pong GEMM: an operand pinned with ``spmw.shared(double=True)`` (M4, plan.md §6.7).

``spmw.shared(B, double=True)`` requests ping-pong double buffering. The rolled HLS emitter lowers a
``double=True`` placement to a two-epoch K-tiled GEMM top with two physical on-chip copies of a K-tile
of B (``B_buf[2][KT][N]``, partitioned on the ping/pong axis), an ``epoch & 1`` selector, and separate
preload/consume loops -- the *next* epoch's tile is preloaded into the alternate copy while the
current copy is consumed, and C is accumulated across the two epochs. A pure systolic streaming GEMM
has no epoch boundary to double-buffer, so a ``double=True`` placement lowers to this ping-pong top
rather than the streaming systolic one (and an unsupported shape fails closed). It still computes
``A @ B``, so the simulator/csim oracle stays ``A @ B``; the tests below assert the ping-pong
*structure*, not only numerical transparency.
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


def _gemm(M, N, K, double=None):
    """A systolic GEMM twin that optionally pins an operand to a ping-pong double buffer."""
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
        if double:
            spmw.place(double, spmw.shared(float32, space="L2", double=True))

    return gemm


def test_shared_double_carries_the_ping_pong_flag():
    assert spmw.shared(float32, double=True).double is True
    assert spmw.shared(float32).double is False  # default is single-buffered


def test_pingpong_placement_round_trips_on_the_rolled_ir():
    """A ``double=True`` placement rides the rolled IR on the ``spmw.double_buffers`` string list."""
    ir = str(spmw.lower(_gemm(4, 4, 4, double="B")))
    match = re.search(r"spmw\.double_buffers = \[([^\]]*)\]", ir)
    assert match and '"B"' in match.group(1)
    # a single-buffered map carries no such attribute
    assert "spmw.double_buffers" not in str(spmw.lower(_gemm(4, 4, 4)))


def test_pingpong_emitter_generates_real_two_copy_ping_pong():
    """The rolled emitter lowers a double-buffered B to a genuine two-copy ping-pong GEMM top.

    Not a dual-port pragma on the top-level operand: the generated C++ must contain two physical
    copies (``B_buf[2][KT][N]``), an ``epoch & 1`` selector, and separate preload/consume loops.
    """
    cpp = emit_rolled_hls_ir(_gemm(4, 4, 4, double="B"))
    assert "B_buf[2][KT][N]" in cpp  # two physical copies (ping + pong)
    assert "#pragma HLS array_partition variable=B_buf complete dim=1" in cpp
    assert (
        "int cur = e & 1" in cpp and "int nxt = (e + 1) & 1" in cpp
    )  # alternating selector
    assert "for (int e = 0; e < 2" in cpp  # two-epoch K tiling
    assert (
        "B_buf[nxt]" in cpp and "B_buf[cur]" in cpp
    )  # preload next while consuming current
    assert "acc[i][j] += s;" in cpp  # accumulation across epochs
    # the Round-5 facade (a RAM_T2P bind on the fully-partitioned top-level operand) is gone
    assert "RAM_T2P" not in cpp
    # a single-buffered map is the streaming systolic top (no ping-pong buffer)
    plain = emit_rolled_hls_ir(_gemm(4, 4, 4))
    assert "B_buf" not in plain and "fb[M + 1][N]" in plain


def test_pingpong_unsupported_shapes_fail_closed():
    """A double placement the ping-pong schedule cannot cover raises, rather than pretending."""
    with pytest.raises(NotImplementedError, match="B operand only"):
        emit_rolled_hls_ir(_gemm(4, 4, 4, double="A"))  # only B is supported
    with pytest.raises(NotImplementedError, match="two epochs"):
        emit_rolled_hls_ir(_gemm(4, 4, 3, double="B"))  # K not divisible into 2 epochs


def test_pingpong_gemm_simulator_matches_numpy():
    """L1: the ping-pong GEMM (double-buffered B) is simulator-correct against numpy A @ B."""
    M, N, K = 4, 4, 8
    np.random.seed(0)
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    sim_mod = spmw.build(_gemm(M, N, K, double="B"), target="simulator")
    sim_mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-4)


@pytest.mark.skipif(
    not hls.is_available("vitis_hls"), reason="requires the Vitis HLS toolchain"
)
def test_pingpong_gemm_rolled_csim():
    """L2: the two-copy ping-pong GEMM top csims correct on Vitis HLS vs A @ B."""
    with tempfile.TemporaryDirectory() as tmp:
        spmw.build(
            _gemm(4, 4, 4, double="B"), target="rolled", project=tmp, testbench=True
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


@pytest.mark.skipif(
    not hls.is_available("vitis_hls"), reason="requires the Vitis HLS toolchain"
)
def test_pingpong_gemm_rolled_csynth():
    """L3: the ping-pong GEMM top synthesizes and its two-copy ping-pong buffer is a real RAM."""
    with tempfile.TemporaryDirectory() as tmp:
        spmw.build(_gemm(4, 4, 4, double="B"), target="rolled", project=tmp)
        subprocess.run(
            ["vitis_hls", "-f", "run.tcl"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=1200,
            check=True,
        )
        report = os.path.join(
            tmp, "rolled.prj", "solution1", "syn", "report", "top_csynth.rpt"
        )
        assert os.path.exists(report), "csynth did not produce a top report"
        text = open(report, encoding="utf-8").read()
    # the ping-pong buffer synthesizes as a real storage (its two copies show up as memory/registers)
    assert "B_buf" in text
