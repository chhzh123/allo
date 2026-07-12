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
import sys

import numpy as np
import pytest

import allo.spmw as spmw
from allo.backend.perf import (
    analyze_sdf,
    analyze_fft_sdf,
    token_clock,
    estimate_area_latency,
    load_csynth_report,
    ResourceVector,
    RoleArea,
    _parse_fft_report_table,
)
from allo.spmw_rollsim import SPMWDeadlockError
from allo.ir.types import float32

sys.path.insert(0, os.path.dirname(__file__))
from test_fft import _fft_region  # noqa: E402 (sibling FFT region builder)

_REPORTS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "examples", "spmw_generated"
)
_FFT_REPORT = os.path.join(_REPORTS, "fft_rolled_csynth_report.md")
_SYSTOLIC_REPORT = os.path.join(_REPORTS, "systolic_rolled_perf_report.md")
_SYSTOLIC_16_REPORT = os.path.join(_REPORTS, "systolic_rolled_16x16_perf_report.md")
_MINI_TPU_REPORT = os.path.join(_REPORTS, "mini_tpu_csynth_report.md")
_U280_DSP = 9024  # DSP slices on the Alveo U280 (the 64x64 systolic overruns this)


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


def test_systolic_area_scale_invariant_extrapolates_to_64x64():
    """The per-role areas are proven **scale-invariant** across the 8×8 and 16×16 archived csynths, so
    `Σ(role_area × instances)` extrapolates exactly to the plan's 64×64 array. The 16×16 area law is
    DSP-exact (256·5 = 1280) with FF/LUT within tolerance, and the 64×64 model predicts 4096·5 = 20480
    DSP -- which exceeds one U280's 9024 DSP: the analytic model forecasts the resource wall the 64×64
    systolic hits, without synthesis."""
    ev8 = load_csynth_report(_read(_SYSTOLIC_REPORT))
    ev16 = load_csynth_report(_read(_SYSTOLIC_16_REPORT))
    area8 = {r.name: r.area for r in ev8.roles}
    area16 = {r.name: r.area for r in ev16.roles}
    # every per-role area is byte-identical across grids -- the O(#roles) scale-invariance.
    assert set(area8) == {"pe_interior", "load_a", "load_b", "drain"}
    assert area8 == area16
    # the area law holds at 16×16: DSP exact, FF/LUT within the top-level-glue tolerance.
    est16 = estimate_area_latency(ev16.roles, latency=0)
    assert est16.total_area.dsp == ev16.top.dsp == 1280  # 256·5, exact
    assert est16.area_within(ev16.top, 0.15)  # DSP 0% / FF ~2% / LUT ~12% -> within 15%
    # extrapolate to 64×64 with the SAME (scale-invariant) per-role areas and 64×64 instance counts.
    counts64 = {"pe_interior": 64 * 64, "load_a": 64, "load_b": 64, "drain": 2 * 64}
    roles64 = [
        RoleArea(name=n, area=area16[n], instances=counts64[n]) for n in counts64
    ]
    est64 = estimate_area_latency(roles64, latency=0)
    assert est64.total_area.dsp == 64 * 64 * 5 == 20480  # exact by scale-invariance
    assert (
        est64.total_area.dsp > _U280_DSP
    )  # 64×64 systolic overruns one U280 (model predicts it)


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


def test_fft_first_class_csynth_evidence():
    """The folded-FFT report loads into the SAME ``CsynthEvidence`` schema as the systolic/Mini-TPU:
    top resources **and** latency/II **and** per-role areas -- the FFT is a first-class evidence source,
    not DSP-only. ``load_csynth_report`` is the identical loader used for the systolic report.
    """
    ev = load_csynth_report(_read(_FFT_REPORT))
    assert ev.top.dsp == 44 and ev.top.ff == 10372 and ev.top.lut == 5729
    assert (
        ev.latency == 91 and ev.ii == 92
    )  # machine-parseable latency AND II (not DSP-only)
    assert {"bfly", "bfly_1", "stage"} <= {r.name for r in ev.roles}


def test_area_latency_matches_fft_csynth():
    """``area = Σ(role_area × instances)`` + an S-stage latency reconstruct the **actual** folded FFT
    csynth: DSP exactly (compute = the two butterfly bodies), FF/LUT within tolerance, and the S-stage
    latency within tolerance of the top latency (91) -- validating latency/area, not DSP-only.
    """
    ev = load_csynth_report(_read(_FFT_REPORT))
    compute = [r for r in ev.roles if r.name.startswith("bfly")]
    stage = next(r for r in ev.roles if r.name == "stage")
    # the folded FFT runs its S stages sequentially: modeled latency = Σ per-stage-loop latency.
    est = estimate_area_latency(ev.roles, latency=stage.instances * stage.latency)
    assert est.total_area.dsp == ev.top.dsp == 44  # 1*24 + 1*20 = 44, exact
    assert (
        sum(r.area.dsp * r.instances for r in compute) == 44
    )  # all compute is the butterflies
    assert est.area_within(ev.top, 0.20)  # DSP 0% / FF ~9.5% / LUT ~16% -> within 20%
    assert est.latency_within(ev.latency, 0.20)  # 3*26 = 78 vs 91 -> ~14%


def test_fft_sdf_folded_structure_matches_report():
    """``analyze_fft_sdf`` models the folded FFT (S=log2(N) sequential stages, HALF butterflies/stage,
    time-shared onto a CONSTANT physical-PE count at II=1) and its structure matches the archived csynth:
    n_stages == the report's stage-loop instance count; physical PEs constant as N scales (O(#roles),
    matching the constant DSP); butterfly II == 1 (the report's ``bfly`` II)."""
    ev = load_csynth_report(_read(_FFT_REPORT))
    sdf8 = analyze_fft_sdf(_fft_region(8, fold={1: 4}))
    stage = next(r for r in ev.roles if r.name == "stage")
    assert (
        sdf8.n_stages == 3 == stage.instances
    )  # analytic stages == report stage-loop instances
    assert sdf8.butterflies == 3 * 4 and sdf8.butterfly_ii == 1
    # full fold -> ONE physical butterfly PE, constant as N scales (the O(#roles) win -> constant DSP).
    sdf16 = analyze_fft_sdf(_fft_region(16, fold={1: 8}))
    assert sdf8.physical_pes == sdf16.physical_pes == 1 and sdf16.n_stages == 4
    rows = _parse_fft_report_table(_read(_FFT_REPORT))
    assert (
        rows[("folded", 8)]["DSP"] == rows[("folded", 16)]["DSP"]
    )  # constant compute == constant PEs


def test_fft_sdf_rejects_non_fft():
    """Fail-closed: ``analyze_fft_sdf`` rejects the systolic mesh, and ``analyze_sdf`` rejects the FFT --
    each analytic model validates only the structure it recognizes (no unvalidated numbers).
    """
    with pytest.raises(NotImplementedError):
        analyze_fft_sdf(_systolic(2, 2, 2))
    with pytest.raises(NotImplementedError):
        analyze_sdf(_fft_region(8, fold={1: 4}))


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
