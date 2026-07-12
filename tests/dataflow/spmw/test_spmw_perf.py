# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPMW analytic perf/area model (M5 task5.2 + task5.3, plan.md M5).

``allo.backend.perf`` estimates a region's performance/area without synthesis. Tier-1
``analyze_sdf(region)`` gives the synchronous-dataflow quantities (firing rates, min FIFO depth,
steady-state II, latency, deadlock threshold); Tier-2 ``token_clock(region)`` runs a timed dataflow
(virtual token clock) for a cycle-accurate latency; ``estimate_area`` computes ``Σ(role_area ×
instances)``. These tests pin each to ground truth: the SDF cycle count equals the round-14 coroutine
simulator's *measured* cycles (and is depth-invariant above the min-depth, exercising the resolved
``port_depths``); the analytic min-depth matches the simulator's real deadlock threshold; the token
clock latency equals the Tier-1 ``hw_latency`` across sizes (incl. 64x64); and the area model
reconstructs the archived folded-FFT csynth DSP exactly (O(#roles), scale-invariant per-role), with a
negative out-of-tolerance case.
"""

import os

import numpy as np
import pytest

import allo.spmw as spmw
from allo.backend.perf import (
    analyze_sdf,
    token_clock,
    estimate_area,
    _parse_fft_report_table,
)
from allo.spmw_rollsim import SPMWDeadlockError
from allo.ir.types import float32

_FFT_REPORT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "examples",
    "spmw_generated",
    "fft_rolled_csynth_report.md",
)


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


def test_token_clock_latency_matches_hw_latency():
    """Tier-2 token clock (timed dataflow) latency equals the Tier-1 analytic ``hw_latency`` across
    systolic sizes -- including the 64x64 array the plan calls out -- an independent cross-check
    (the token clock runs the wavefront recurrence; the SDF model reports a closed form).
    """
    for M, N, K in [(2, 2, 2), (3, 3, 3), (2, 3, 4), (4, 4, 4), (8, 8, 8), (64, 64, 8)]:
        tc = token_clock(_systolic(M, N, K))
        sdf = analyze_sdf(_systolic(M, N, K))
        assert tc.latency == sdf.hw_latency, (M, N, K, tc.latency, sdf.hw_latency)
        assert tc.dims == (M, N, K) and tc.throughput_ii == 1


def test_area_model_folded_fft_is_o_roles():
    """``area = Σ(role_area × instances)`` with a scale-invariant per-role area: the folded FFT's
    compute area (DSP) predicted from the N=8 role area matches the N=16 csynth exactly (constant =
    O(#roles)), while the spatial FFT's DSP grows with the body count (O(P))."""
    with open(_FFT_REPORT, encoding="utf-8") as handle:
        rows = _parse_fft_report_table(handle.read())
    f8, f16 = rows[("folded", 8)], rows[("folded", 16)]
    # per-role (per-body) compute area from the small case; scale to the N=16 body count
    per_role = {"DSP": f8["DSP"] // f8["bodies"]}
    est = estimate_area(per_role, f16["bodies"])
    assert (
        est.total["DSP"] == f16["DSP"]
    )  # folded compute area is constant across N (44)
    assert est.within(f16["DSP"], "DSP", 0.05)
    # the folded body count (roles) is constant while the spatial body count grows (O(P))
    assert f8["bodies"] == f16["bodies"]
    assert rows[("spatial", 16)]["bodies"] > rows[("spatial", 8)]["bodies"]
    assert rows[("spatial", 16)]["DSP"] > rows[("spatial", 8)]["DSP"]


def test_area_model_rejects_out_of_tolerance():
    """Negative: a wrong per-role area makes the ``Σ(role_area × instances)`` estimate miss the
    archived total -- the tolerance check fails instead of silently accepting it."""
    with open(_FFT_REPORT, encoding="utf-8") as handle:
        rows = _parse_fft_report_table(handle.read())
    f16 = rows[("folded", 16)]
    bad = estimate_area(
        {"DSP": 2 * (f16["DSP"] // f16["bodies"])}, f16["bodies"]
    )  # 2x too big
    assert bad.total["DSP"] == 2 * f16["DSP"]
    assert not bad.within(f16["DSP"], "DSP", 0.10)


def test_token_clock_rejects_non_mesh():
    """The token clock is fail-closed on patterns it has not validated (a channel pipeline)."""

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
        token_clock(top)
