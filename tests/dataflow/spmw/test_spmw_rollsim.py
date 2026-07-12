# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPMW rolled coroutine simulator (M5 task5.1, plan.md M5).

``spmw.build(region, target="rolled_simulator")`` runs a region on a cooperative coroutine scheduler --
one generator per logical PID, bounded-FIFO ``put``/``get``, single-threaded round-robin -- instead of
the desugar ``target="simulator"`` path that lowers to ``allo.dataflow`` and injects one OpenMP section
per PE. These tests prove it reproduces the desugar simulator (and numpy) on the 2-D systolic mesh and
the 1-D producer/consumer pipeline, needs no ``OMP_NUM_THREADS``, and *detects* a stuck schedule
(reporting the blocked coroutines) rather than hanging a host thread.
"""

import os

import numpy as np
import pytest

import allo
import allo.spmw as spmw
from allo.ir.types import float32


def _systolic(M, N, K):
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


def _producer_consumer(M, N, over_read=False):
    @spmw.unit
    def producer(ctx):
        for i, j in allo.grid(M, N):
            out: float32 = ctx.A[i, j]
            ctx.pipe.put(out)

    if not over_read:

        @spmw.unit
        def consumer(ctx):
            for i, j in allo.grid(M, N):
                data: float32 = ctx.pipe.get()
                ctx.B[i, j] = data + 1

    else:  # reads one more token than the producer puts -> the last get blocks forever (deadlock)

        @spmw.unit
        def consumer(ctx):
            for i, j in allo.grid(M, N):
                data: float32 = ctx.pipe.get()
                ctx.B[i, j] = data + 1
            _extra: float32 = ctx.pipe.get()

    @spmw.region()
    def top(A: float32[M, N], B: float32[M, N]):
        spmw.map(producer, grid=(1,))
        spmw.map(consumer, grid=(1,))
        spmw.channel("pipe", float32, depth=4)

    return top


def test_rollsim_systolic_matches_numpy_and_desugar():
    """The coroutine simulator matches numpy ``A @ B`` and the desugar ``target="simulator"`` on a
    2-D systolic mesh (a non-square shape so the halo index math is genuinely exercised).
    """
    M, N, K = 3, 4, 5
    np.random.seed(0)
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    roll = np.zeros((M, N), dtype=np.float32)
    spmw.build(_systolic(M, N, K), target="rolled_simulator")(A, B, roll)
    np.testing.assert_allclose(roll, A @ B, atol=1e-4)
    desugar = np.zeros((M, N), dtype=np.float32)
    spmw.build(_systolic(M, N, K), target="simulator")(A, B, desugar)
    np.testing.assert_allclose(roll, desugar, atol=1e-5)


def test_rollsim_producer_consumer_matches():
    """The coroutine simulator matches ``A + 1`` and the desugar simulator on a 1-D channel pipeline."""
    M, N = 4, 4
    np.random.seed(1)
    A = np.random.rand(M, N).astype(np.float32)
    roll = np.zeros((M, N), dtype=np.float32)
    spmw.build(_producer_consumer(M, N), target="rolled_simulator")(A, roll)
    np.testing.assert_allclose(roll, A + 1, atol=1e-5)
    desugar = np.zeros((M, N), dtype=np.float32)
    spmw.build(_producer_consumer(M, N), target="simulator")(A, desugar)
    np.testing.assert_allclose(roll, desugar, atol=1e-6)


def test_rollsim_has_no_omp_num_threads_dependency():
    """The coroutine simulator reads no ``OMP_NUM_THREADS``: it runs correctly with the var unset."""
    M, N, K = 3, 3, 4
    np.random.seed(2)
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    saved = os.environ.pop("OMP_NUM_THREADS", None)
    try:
        spmw.build(_systolic(M, N, K), target="rolled_simulator")(A, B, C)
    finally:
        if saved is not None:
            os.environ["OMP_NUM_THREADS"] = saved
    np.testing.assert_allclose(C, A @ B, atol=1e-4)


def test_rollsim_detects_deadlock_not_hang():
    """An unsatisfiable schedule (a consumer that reads one more token than is produced) is detected as
    a deadlock -- naming the blocked coroutines -- not left to hang a host thread."""
    from allo.spmw_rollsim import SPMWDeadlockError

    M, N = 2, 2
    A = np.random.rand(M, N).astype(np.float32)
    B = np.zeros((M, N), dtype=np.float32)
    with pytest.raises(SPMWDeadlockError, match="deadlock"):
        spmw.build(_producer_consumer(M, N, over_read=True), target="rolled_simulator")(
            A, B
        )
