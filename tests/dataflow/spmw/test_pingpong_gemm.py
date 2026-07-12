# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPMW ping-pong GEMM: an operand pinned with ``spmw.shared(double=True)`` (M4, plan.md §6.7).

``spmw.shared(B, double=True)`` requests ping-pong double buffering. The rolled HLS emitter lowers a
``double=True`` placement to a two-epoch K-tiled GEMM top with two physical on-chip copies of a K-tile
of B (``B_buf[2][KT][N]``, partitioned on the ping/pong axis). The load and compute over the two
copies are emitted as named stage functions (``load_b_tile``/``compute_b_tile``) invoked as **sibling
tasks inside a** ``#pragma HLS dataflow`` **region**, so the load of one copy structurally overlaps
the compute of the other, and the two epoch partials are summed into C. It still computes ``A @ B``,
so the simulator/csim oracle stays ``A @ B``; the tests assert the ping-pong *structure* (overlap +
two copies), that the branch is *role-faithful* (a non-MAC PE fails closed), and that it does not
silently drop other placements -- not only numerical transparency. A pure systolic streaming GEMM has
no epoch boundary to double-buffer, so ``double=True`` lowers to this tiled ping-pong top.
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


def _gemm(M, N, K, double=None, variant="mac", also_place=None):
    """A systolic GEMM twin, optionally double-buffering an operand and/or with a non-MAC PE body."""
    grid = spmw.mesh((M, N))

    if variant == "mac":

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

    else:  # a non-canonical PE (an extra add on top of the MAC): must fail closed under ping-pong

        @spmw.unit
        def pe(ctx):
            c: float32 = 0
            for k in range(K):
                a: float32 = ctx.west.get()
                b: float32 = ctx.north.get()
                c += a * b + a
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
        if also_place:
            spmw.place(also_place, spmw.shared(float32, resource="URAM"))

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


def test_pingpong_emitter_generates_overlapped_two_copy_ping_pong():
    """The rolled emitter lowers a double-buffered B to a real, *overlapped* two-copy ping-pong top.

    Named ``load_b_tile``/``compute_b_tile`` stage functions run as sibling tasks inside a
    ``#pragma HLS dataflow`` region over two physical copies (``B_buf[2][KT][N]``) -- so the load of
    one copy overlaps the compute of the other, rather than two sequential loops or a dual-port pragma.
    """
    cpp = emit_rolled_hls_ir(_gemm(4, 4, 4, double="B"))
    assert "B_buf[2][KT][N]" in cpp  # two physical copies (ping + pong)
    assert "#pragma HLS array_partition variable=B_buf complete dim=1" in cpp
    assert "#pragma HLS dataflow" in cpp  # the overlap region
    assert (
        "void load_b_tile(" in cpp and "void compute_b_tile(" in cpp
    )  # named stage functions
    # the load of one copy and the compute of the other are sibling tasks in the dataflow region
    assert "load_b_tile(0, B, B_buf[0]);" in cpp
    assert "compute_b_tile(0, A, B_buf[0], acc0);" in cpp
    assert "load_b_tile(1, B, B_buf[1]);" in cpp
    assert "compute_b_tile(1, A, B_buf[1], acc1);" in cpp
    assert "C[i][j] = acc0[i][j] + acc1[i][j];" in cpp  # sum the two epoch partials
    # the Round-5/6 facade (a RAM_T2P bind on the fully-partitioned top-level operand) is gone
    assert "RAM_T2P" not in cpp
    # a single-buffered map is the streaming systolic top (no ping-pong buffer / stage functions)
    plain = emit_rolled_hls_ir(_gemm(4, 4, 4))
    assert (
        "B_buf" not in plain and "load_b_tile" not in plain and "fb[M + 1][N]" in plain
    )


def test_pingpong_is_role_faithful_and_fails_closed():
    """The ping-pong branch hard-codes A@B, so it must reject anything but the supported MAC GEMM PE.

    A non-canonical PE (here ``c += a*b + a``), a double placement on A/C, and a K that does not split
    into two epochs all raise -- rather than silently emitting a plain matmul or dropping placements.
    """
    with pytest.raises(NotImplementedError, match="output-stationary MAC"):
        emit_rolled_hls_ir(_gemm(4, 4, 4, double="B", variant="bias"))
    # the same non-MAC PE is fine WITHOUT double buffering (the systolic top transcribes it)
    emit_rolled_hls_ir(_gemm(4, 4, 4, variant="bias"))
    with pytest.raises(NotImplementedError, match="B operand only"):
        emit_rolled_hls_ir(_gemm(4, 4, 4, double="A"))  # only B is supported
    with pytest.raises(NotImplementedError, match="two epochs"):
        emit_rolled_hls_ir(_gemm(4, 4, 3, double="B"))  # K not divisible into 2 epochs
    with pytest.raises(NotImplementedError, match="does not honor placements"):
        emit_rolled_hls_ir(_gemm(4, 4, 4, double="B", also_place="A"))  # A placed too


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
def test_pingpong_gemm_rolled_csynth_overlapped():
    """L3: the dataflow ping-pong top synthesizes load and compute as distinct concurrent modules."""
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
        reports = [
            os.path.basename(p)
            for p in os.scandir(
                os.path.join(tmp, "rolled.prj", "solution1", "syn", "report")
            )
        ]
    # the load and compute stages synthesize as their own (sibling, dataflow) modules
    assert any("load_b_tile" in r for r in reports), reports
    assert any("compute_b_tile" in r for r in reports), reports
