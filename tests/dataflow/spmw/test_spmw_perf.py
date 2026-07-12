# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPMW analytic perf/area model (M5 task5.2 + task5.3, plan.md M5).

``allo.backend.perf`` estimates a region's performance/area without synthesis. Tier-1
``analyze_sdf(region)`` gives the synchronous-dataflow quantities (firing rates, min FIFO depth,
steady-state II, latency, deadlock threshold); Tier-2 ``token_clock(region)`` runs a timed dataflow
(virtual token clock) for a cycle-accurate latency; ``estimate_area_latency`` computes
``Σ(role_area × instances)`` + a modeled latency over per-role module areas loaded from the archived
csynth reports. These tests pin each to ground truth: the SDF cycle count equals the round-14 coroutine
simulator's *measured* cycles (depth-invariant above the min-depth); the analytic min-depth matches the
simulator's real deadlock threshold; the token clock latency equals the Tier-1 ``hw_latency`` across
sizes (incl. 64x64); and the area+latency model reconstructs the **actual** archived systolic (DSP
exact, FF/LUT within a glue tolerance, load+PE latency vs 79), Mini-TPU (multi-role, DSP exact), and
folded-FFT (O(#roles)) csynth numbers -- with a negative out-of-tolerance case and fail-closed
non-mesh.
"""

import os

import numpy as np
import pytest

import allo.spmw as spmw
from allo.backend.perf import (
    analyze_sdf,
    token_clock,
    estimate_area_latency,
    load_csynth_report,
    ResourceVector,
    RoleArea,
    _parse_fft_report_table,
)
from allo.spmw_rollsim import SPMWDeadlockError
from allo.ir.types import float32

_REPORTS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "examples", "spmw_generated"
)
_FFT_REPORT = os.path.join(_REPORTS, "fft_rolled_csynth_report.md")
_SYSTOLIC_REPORT = os.path.join(_REPORTS, "systolic_rolled_perf_report.md")
_MINI_TPU_REPORT = os.path.join(_REPORTS, "mini_tpu_csynth_report.md")


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


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


def test_area_latency_matches_systolic_csynth():
    """``area = Σ(role_area × instances)`` and a modeled latency reconstruct the **actual** archived
    systolic rolled csynth (8x8): DSP exactly, FF/LUT within a top-level-glue tolerance, and the
    load+PE critical-path latency within tolerance of the csynth latency (79)."""
    ev = load_csynth_report(_read(_SYSTOLIC_REPORT))
    assert ev.top.dsp == 320 and ev.top.ff == 44690 and ev.top.lut == 38948
    assert len(ev.roles) == 4  # pe_interior, load_a, load_b, drain
    load_lat = max(r.latency for r in ev.roles if r.name.startswith("load"))
    pe_lat = next(r.latency for r in ev.roles if r.name == "pe_interior")
    est = estimate_area_latency(ev.roles, latency=load_lat + pe_lat)
    assert est.total_area.dsp == ev.top.dsp  # 64 x 5 = 320, exact
    assert est.area_within(ev.top, 0.15)  # DSP 0% / FF ~2% / LUT ~13% -> within 15%
    assert est.latency_within(ev.latency, 0.20)  # load(10)+PE(64)=74 vs 79 -> ~6%


def test_area_model_matches_mini_tpu_csynth_multi_role():
    """A multi-role summation over the Mini-TPU's distinct module classes (mxu / act / load_buf /
    store_res): DSP matches the top exactly; FF is within the (interconnect-heavy) tolerance.
    """
    ev = load_csynth_report(_read(_MINI_TPU_REPORT))
    assert {r.name for r in ev.roles} == {"mxu", "act", "load_buf", "store_res"}
    est = estimate_area_latency(ev.roles, latency=0)
    assert est.total_area.dsp == ev.top.dsp == 112  # 16 x 5 + 4 x 8 = 112, exact
    assert (
        abs(est.total_area.ff - ev.top.ff) <= 0.30 * ev.top.ff
    )  # ~22% (dataflow glue)


def test_area_model_folded_fft_is_o_roles():
    """The folded FFT's compute area (DSP) predicted from the N=8 per-role area x the (constant) body
    count matches the N=16 csynth exactly -- O(#roles), scale-invariant -- while spatial DSP grows.
    """
    rows = _parse_fft_report_table(_read(_FFT_REPORT))
    f8, f16 = rows[("folded", 8)], rows[("folded", 16)]
    per_role = ResourceVector(dsp=f8["DSP"] // f8["bodies"])
    est16 = estimate_area_latency(
        [RoleArea("bfly", per_role, f16["bodies"])], latency=0
    )
    assert (
        est16.total_area.dsp == f16["DSP"]
    )  # 2 x 22 = 44, constant across N (O(#roles))
    assert f8["bodies"] == f16["bodies"]
    assert rows[("spatial", 16)]["DSP"] > rows[("spatial", 8)]["DSP"]  # spatial is O(P)


def test_area_latency_model_rejects_out_of_tolerance():
    """Negative: a wrong per-role area AND a wrong latency both fail the tolerance checks against the
    archived systolic report -- an out-of-tolerance estimate is rejected, not accepted.
    """
    ev = load_csynth_report(_read(_SYSTOLIC_REPORT))
    doubled = [
        RoleArea(
            name=r.name,
            area=r.area.scale(2) if r.name == "pe_interior" else r.area,
            instances=r.instances,
            latency=r.latency,
        )
        for r in ev.roles
    ]
    bad = estimate_area_latency(doubled, latency=5 * ev.latency)
    assert not bad.area_within(ev.top, 0.15)  # doubled PE area -> DSP ~640 vs 320
    assert not bad.latency_within(ev.latency, 0.20)  # 5x latency


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
