# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import tempfile

import numpy as np
import pytest
import allo.spmw as spmw
import allo.backend.hls as hls
from allo.ir.types import float32


def _systolic_twin(M, N, K):
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

    return gemm


def test_systolic_twin_emits_vitis_hls_code():
    # the desugared region produces synthesizable HLS C++ (no toolchain needed to inspect it)
    with tempfile.TemporaryDirectory() as tmp:
        module = spmw.build(_systolic_twin(2, 2, 2), target="vitis_hls", project=tmp)
    assert "void" in module.hls_code
    assert "hls::stream" in module.hls_code


def _per_pid_body_count(M, N, K):
    with tempfile.TemporaryDirectory() as tmp:
        module = spmw.build(_systolic_twin(M, N, K), target="vitis_hls", project=tmp)
    return len(set(re.findall(r"gemm_\d+_\d+", module.hls_code)))


def test_synthesis_time_win_df_grows_but_rolled_stays_constant():
    # the df-desugaring path clones one HLS function body per grid point (the O(P0*P1) blow-up):
    # a bigger grid emits strictly more per-PID `gemm_i_j` bodies for the synthesizer to schedule.
    assert _per_pid_body_count(6, 6, 4) > _per_pid_body_count(4, 4, 4) > 0
    # the rolled spmw HLS emitter instead stays O(#roles) = 9 regardless of grid size.
    assert spmw.role_body_count(_systolic_twin(4, 4, 4)) == 9
    assert spmw.role_body_count(_systolic_twin(6, 6, 4)) == 9


@pytest.mark.skipif(
    not hls.is_available("vitis_hls"), reason="requires the Vitis HLS toolchain"
)
def test_systolic_twin_vitis_hls_csim():
    M, N, K = 2, 2, 2
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        module = spmw.build(
            _systolic_twin(M, N, K), target="vitis_hls", mode="csim", project=tmp
        )
        module(A, B, C)
    np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)


@pytest.mark.skipif(
    not hls.is_available("vitis_hls"), reason="requires the Vitis HLS toolchain"
)
def test_systolic_twin_csyn_emits_synthesizable_project():
    # mode="csyn" applies complete partitioning so HLS dataflow sees a single writer per bank, and
    # emits a synthesizable Vitis project (csynth itself is run out of band; see round 12 report).
    with tempfile.TemporaryDirectory() as tmp:
        spmw.build(
            _systolic_twin(2, 2, 2), target="vitis_hls", mode="csyn", project=tmp
        )
        assert os.path.exists(os.path.join(tmp, "run.tcl"))
        assert os.path.exists(os.path.join(tmp, "kernel.cpp"))
