# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPMW Tier-1 analytic SDF performance model (M5 task5.2, plan.md M5).

``allo.backend.perf.analyze_sdf(region)`` statically estimates the synchronous-dataflow quantities of a
region (firing rates, min FIFO depth, steady-state II, latency, deadlock threshold). These tests pin it
to ground truth: its predicted cycle count equals the round-14 coroutine simulator's *measured* cycle
count on the systolic mesh across sizes (and is depth-invariant above the min-depth); its analytic
min-depth matches the simulator's actual deadlock threshold (depth 0 deadlocks, depth >= 1 runs). The
rollsim now sizes its mesh FIFOs from the map's resolved ``port_depths`` (not a hard-coded constant),
which the depth-0/-1 cases exercise.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.backend.perf import analyze_sdf
from allo.spmw_rollsim import SPMWDeadlockError
from allo.ir.types import float32


def _systolic(M, N, K, depths=None):
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
        spmw.map(pe, grid=grid, depths=depths)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    return gemm


def _run_rollsim(M, N, K, depths=None):
    np.random.seed(0)
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    cycles = spmw.build(_systolic(M, N, K, depths), target="rolled_simulator")(A, B, C)
    return cycles, C, A @ B


def test_sdf_sim_cycles_match_rollsim_across_sizes():
    """The analytic ``sim_cycles`` equals the coroutine simulator's measured cycle count exactly, over a
    range of (non-square) systolic shapes."""
    for M, N, K in [(2, 2, 2), (3, 3, 3), (2, 3, 4), (4, 4, 4), (3, 4, 5), (2, 2, 8)]:
        report = analyze_sdf(_systolic(M, N, K))
        cycles, out, ref = _run_rollsim(M, N, K)
        assert report.sim_cycles == cycles, (M, N, K, report.sim_cycles, cycles)
        np.testing.assert_allclose(out, ref, atol=1e-4)


def test_sdf_rates_throughput_latency():
    """The analytic firing rate (K per PE), steady-state II (1), min-depth (1), and hardware II=1
    latency (K + M + N - 2) for the output-stationary systolic mesh."""
    M, N, K = 4, 5, 6
    report = analyze_sdf(_systolic(M, N, K))
    assert report.pattern == "systolic_mesh" and report.dims == (M, N, K)
    assert report.role_fires["interior"] == K
    assert report.throughput_ii == 1
    assert report.min_depth == 1
    assert report.hw_latency == K + M + N - 2


def test_sdf_min_depth_matches_rollsim_deadlock_threshold():
    """The analytic min-depth (1) is the real deadlock threshold: a depth-0 FIFO deadlocks the
    simulator, a depth-1 FIFO runs correctly -- and the cycle count is depth-invariant above min-depth,
    proving the rollsim uses the resolved ``port_depths`` (a hard-coded 4 could never deadlock).
    """
    M, N, K = 2, 2, 3
    report = analyze_sdf(_systolic(M, N, K))
    assert report.deadlocks_at(0) and not report.deadlocks_at(1)
    zero = {p: 0 for p in ("west", "east", "north", "south")}
    with pytest.raises(SPMWDeadlockError, match="deadlock"):
        _run_rollsim(M, N, K, depths=zero)
    one = {p: 1 for p in ("west", "east", "north", "south")}
    cycles, out, ref = _run_rollsim(M, N, K, depths=one)
    np.testing.assert_allclose(out, ref, atol=1e-4)
    assert cycles == report.sim_cycles  # depth-invariant above min-depth


def test_sdf_rejects_non_mesh():
    """The Tier-1 model is fail-closed: a channel pipeline (not the systolic mesh) is not silently
    given an unvalidated number."""

    @spmw.unit
    def producer(ctx):
        ctx.pipe.put(ctx.A[0])

    @spmw.unit
    def consumer(ctx):
        ctx.B[0] = ctx.pipe.get()

    @spmw.region()
    def top(A: float32[1], B: float32[1]):
        spmw.map(producer, grid=(1,))
        spmw.map(consumer, grid=(1,))
        spmw.channel("pipe", float32, depth=2)

    with pytest.raises(NotImplementedError, match="systolic mesh"):
        analyze_sdf(top)
