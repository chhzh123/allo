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

import glob
import os
import re
import subprocess
import tempfile
from math import log2, cos, sin, pi

import numpy as np
import pytest

import allo.spmw as spmw
import allo.backend.hls as hls
from allo.spmw_hls import emit_rolled_hls_ir
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


def _fft_region(N, fold=None):
    """An N-point FFT as an SPMW region (butterfly unit + key-form ``lane`` topology).

    ``fold`` (e.g. ``{1: N // 2}``) time-multiplexes the butterfly axis onto fewer physical PEs; a
    folded map's ``lane`` families reclassify from FIFO streams into addressed buffers.
    """
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
    def fft(
        Xr: float32[N],
        Xi: float32[N],
        Yr: float32[N],
        Yi: float32[N],
    ):
        spmw.map(bfly, grid=(S, HALF), topo=topo, fold=fold)
        spmw.stream_in(
            (Xr, Xi), into=("lane_re", "lane_im"), at_stage=0, index=bit_reverse
        )
        spmw.stream_out((Yr, Yi), from_=("lane_re", "lane_im"), at_stage=S)

    return fft


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


@pytest.mark.skipif(
    not hls.is_available("vitis_hls"), reason="requires the Vitis HLS toolchain"
)
@pytest.mark.parametrize("N", [8])
def test_fft_csim(N):
    """The SPMW spatial FFT is csim-correct on Vitis HLS (fft_spatial at L2 csim vs numpy).

    Builds the desugared FFT to a Vitis HLS project and runs csim; the emitted C++ butterfly
    network must produce numpy.fft.fft to float tolerance -- the L2 rung above the L1 simulator
    check in ``test_fft_spatial``.
    """
    np.random.seed(42)
    inp_real = np.random.rand(N).astype(np.float32)
    inp_imag = np.zeros(N, dtype=np.float32)
    out_real = np.zeros(N, dtype=np.float32)
    out_imag = np.zeros(N, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        module = spmw.build(
            _fft_region(N), target="vitis_hls", mode="csim", project=tmp
        )
        module(inp_real, inp_imag, out_real, out_imag)
    ref = np.fft.fft(inp_real + 1j * inp_imag)
    np.testing.assert_allclose(
        out_real, ref.real.astype(np.float32), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        out_imag, ref.imag.astype(np.float32), rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("N", [8, 16])
def test_fft_rolled_emits_single_butterfly_body(N):
    """The rolled key-form FFT top has ONE bfly compute body regardless of grid size (O(#roles)).

    The key-form ``lane`` topology survives the rolled O(#roles) HLS emitter: the butterfly datapath
    is transcribed once (twiddle lifted to parameters) and instantiated across the (stage, butterfly)
    grid, so the emitted top has a single ``void bfly(`` body while the number of *instantiations*
    grows with the grid.
    """
    cpp = emit_rolled_hls_ir(_fft_region(N))
    assert cpp.count("void bfly(") == 1  # one compute body, not one per butterfly
    assert "stage_lane_re[S + 1][N]" in cpp and "stage_lane_im[S + 1][N]" in cpp
    assert "#pragma HLS dataflow" in cpp
    # the body count is constant, but the number of butterfly instantiations scales with the grid
    assert cpp.count("  bfly(") == int(log2(N)) * (N // 2)


@pytest.mark.skipif(
    not hls.is_available("vitis_hls"), reason="requires the Vitis HLS toolchain"
)
def test_fft_rolled_csim():
    """The rolled key-form FFT top is csim-correct on Vitis HLS (fft_spatial via target="rolled").

    Unlike ``test_fft_csim`` (the desugared ``vitis_hls`` path), this exercises the rolled O(#roles)
    SPMW HLS emitter: the transcribed butterfly body instantiated over the topology's slot wiring,
    checked against a self-checking DFT testbench.
    """
    with tempfile.TemporaryDirectory() as tmp:
        spmw.build(_fft_region(8), target="rolled", project=tmp, testbench=True)
        assert os.path.exists(os.path.join(tmp, "tb.cpp"))
        result = subprocess.run(
            ["vitis_hls", "-f", "run.tcl"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=900,
            check=True,
        )
    assert "CSIM MATCH" in result.stdout, result.stdout[-2000:]


def test_fft_folded_rolled_uses_buffers():
    """A folded FFT map materializes lane stages as addressed on-chip buffers, not FIFOs.

    ``fold={1: F}`` (``spmw.buffer_families``) makes the rolled top a single folded compute body over
    per-stage register arrays read/written by a pipelined butterfly loop -- no lane ``hls::stream``.
    """
    cpp = emit_rolled_hls_ir(_fft_region(8, fold={1: 4}))
    assert cpp.count("void bfly(") == 1  # one folded compute body
    assert "hls::stream<" not in cpp  # lanes are addressed buffers, not FIFO streams
    assert "stage_lane_re_0[N]" in cpp and "stage_lane_im_0[N]" in cpp
    assert "#pragma HLS array_partition variable=stage_lane_re_0 complete dim=0" in cpp
    assert "#pragma HLS pipeline II=1" in cpp


@pytest.mark.parametrize("F", [2, 4])
def test_fft_folded_rolled_honors_fold_factor(F):
    """fold[1]=F emits HALF/F parallel butterfly PEs x F pipelined iterations.

    ``fold[1]=F`` runs F logical butterflies per physical PE, so the butterfly axis (extent HALF)
    becomes HALF/F unrolled PEs each looping F times. Partial (F=2) and full (F=4) folds therefore
    emit distinct schedules -- the round-2 review found the fold factor was parsed but ignored, so
    both produced identical code.
    """
    S, HALF = int(log2(8)), 8 // 2
    cpp = emit_rolled_hls_ir(_fft_region(8, fold={1: F}))
    assert re.findall(r"for \(int i = 0; i < (\d+); i\+\+\)", cpp) == [str(F)] * S
    assert (
        cpp.count("    bfly(") == (HALF // F) * S
    )  # HALF/F parallel PEs, over S stages
    other = HALF if F != HALF else 2
    assert cpp != emit_rolled_hls_ir(_fft_region(8, fold={1: other}))


@pytest.mark.skipif(
    not hls.is_available("vitis_hls"), reason="requires the Vitis HLS toolchain"
)
@pytest.mark.parametrize("F", [2, 4])
def test_fft_folded_rolled_csim(F):
    """The folded key-form FFT (fold -> banked lane buffers) is csim-correct via target="rolled".

    Covers both a partial fold (F=2 -> HALF/2 physical PEs) and a full fold (F=4 -> 1 PE).
    """
    with tempfile.TemporaryDirectory() as tmp:
        spmw.build(
            _fft_region(8, fold={1: F}), target="rolled", project=tmp, testbench=True
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
@pytest.mark.parametrize("F", [2, 4])
def test_fft_folded_rolled_csynth_ii1(F):
    """The folded FFT synthesizes conflict-free banked lane access at II=1 (the AC-6 FFT hard gate).

    Per-stage register arrays keep each butterfly-loop iteration's read (stage s) and write (stage
    s+1) on distinct arrays, so the pipelined folded loops schedule at II=1 for both partial (F=2)
    and full (F=4) folds; and the butterfly is rolled -- far fewer synthesized ``bfly`` bodies than
    the S*HALF butterflies.
    """
    with tempfile.TemporaryDirectory() as tmp:
        spmw.build(_fft_region(8, fold={1: F}), target="rolled", project=tmp)
        subprocess.run(
            ["vitis_hls", "-f", "run.tcl"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=1200,
            check=True,
        )
        report_dir = os.path.join(tmp, "rolled.prj", "solution1", "syn", "report")
        texts = {
            os.path.basename(p): open(p, encoding="utf-8").read()
            for p in glob.glob(os.path.join(report_dir, "*.rpt"))
        }
    # every pipelined butterfly loop achieves II=1. The per-loop pipeline report tabulates each loop
    # as `<name> | latency_min | latency_max | iteration_latency | achieved_II | target_II | count |
    # yes`; the achieved II is the 4th number.
    achieved = {}
    for name, txt in texts.items():
        if not name.startswith("top_Pipeline_VITIS_LOOP"):
            continue
        row = re.search(r"(VITIS_LOOP\S*)\s*((?:\|\s*\d+\s*)+)\|\s*yes", txt)
        if row:
            nums = re.findall(r"\d+", row.group(2))
            if len(nums) >= 4:
                achieved[row.group(1)] = int(nums[3])
    assert achieved and all(
        ii == 1 for ii in achieved.values()
    ), f"folded butterfly loops must be II=1; got {achieved}"
    # rolled: the synthesized butterfly bodies are far fewer than the S*HALF butterflies
    n_bfly = sum(1 for name in texts if name.startswith("bfly"))
    assert 0 < n_bfly < int(log2(8)) * (8 // 2), f"expected a rolled bfly; got {n_bfly}"


@pytest.mark.parametrize(
    "N,fold",
    [
        (8, {1: 2}),
        (8, {1: 4}),
        (16, {1: 2}),
        (16, {1: 8}),
        (32, {1: 4}),
        (32, {1: 16}),
    ],
)
def test_fft_folded(N, fold):
    """A folded SPMW FFT still matches numpy.fft.fft on the simulator (fft_folded sim-correct).

    ``fold={1: F}`` time-multiplexes F butterflies of the N/2-wide axis onto one physical PE, which
    reclassifies each stage's ``lane_re``/``lane_im`` key family from a FIFO into an addressed buffer
    (see ``test_fft_folded_resolve_channels_buffers``). This asserts the *numeric* result survives
    that reclassification -- folding changes the spatial-vs-temporal schedule, not the transform.
    """
    np.random.seed(42)
    inp_real = np.random.rand(N).astype(np.float32)
    inp_imag = np.zeros(N, dtype=np.float32)
    out_real = np.zeros(N, dtype=np.float32)
    out_imag = np.zeros(N, dtype=np.float32)
    sim_mod = spmw.build(_fft_region(N, fold=fold), target="simulator")
    sim_mod(inp_real, inp_imag, out_real, out_imag)
    ref = np.fft.fft(inp_real + 1j * inp_imag)
    np.testing.assert_allclose(
        out_real, ref.real.astype(np.float32), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        out_imag, ref.imag.astype(np.float32), rtol=1e-4, atol=1e-4
    )


def test_fft_spatial_rolled_ir():
    """The SPMW spatial FFT lowers to rolled spmw.map IR carrying key_link lane families."""
    printed = str(spmw.build(_fft_region(8), target="ir"))
    assert "spmw.map" in printed
    assert "key_link" in printed
    assert "lane_re" in printed and "lane_im" in printed


def test_fft_spatial_resolve_channels():
    """spmw-resolve-channels groups the FFT key links into lane_re/lane_im stream families."""
    from allo.spmw import lower, _run_module_pass

    module = lower(_fft_region(8))
    _run_module_pass(module, "spmw-resolve-channels")
    printed = str(module)
    assert "spmw.channel_families" in printed
    assert "lane_re" in printed and "lane_im" in printed


def test_fft_folded_resolve_channels_buffers():
    """A folded FFT map reclassifies its lane_re/lane_im key families into addressed buffers."""
    from allo.spmw import lower, _run_module_pass

    module = lower(_fft_region(8, fold={1: 4}))
    _run_module_pass(module, "spmw-resolve-channels")
    printed = str(module)
    assert "spmw.buffer_families" in printed
    assert "lane_re" in printed and "lane_im" in printed
