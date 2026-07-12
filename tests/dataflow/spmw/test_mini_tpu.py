# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPMW Mini-TPU: hierarchical, heterogeneous composition (M4, plan.md §3.4).

A TPU-style inference tile written as a composition: a **nested** ``@spmw.region()`` MXU (the §3.1
output-stationary systolic mesh, reused as a sub-block) feeds a **distinct** 1-D vector activation
unit (bias + ReLU, one lane per output column) through the ``psum`` buffer. This exercises the three
things §3.4 is about: hierarchy by composition (write the MXU once, instantiate it), heterogeneous
units (a MAC PE and a bias+ReLU lane on different grids), and two connection kinds (``shared``/
``banked`` memory placements for the UB/weights/psum, and streams between the stages).

The composed region desugars to a two-kernel ``allo.dataflow`` program: the systolic MXU streams each
interior PE's psum onto a per-element stream, and the activation lane reads its column in row order,
adds the bias, applies ReLU, and writes ``OUT`` -- so the result matches
``np.maximum(ACT @ WGT + bias, 0)``. This drives the full ladder through the same ``target="vitis_hls"``
path the systolic twin uses: L0 structure, L1 simulator, L2 Vitis csim, L3 csynth (report archived
under ``examples/spmw_generated/mini_tpu_csynth_report.md``), and L4 hw_emu project emission.
"""

import os
import tempfile

import numpy as np
import pytest

import allo.spmw as spmw
import allo.backend.hls as hls
from allo.ir.types import float32


def _mini_tpu(Rt, Ct, K, psum_bank="col"):
    """The §3.4 Mini-TPU: a nested systolic MXU region + a 1-D bias/ReLU activation stage."""

    @spmw.unit
    def pe(
        ctx,
    ):  # the §3.1 output-stationary MAC PE (identical to the systolic/ping-pong twins)
        c: float32 = 0
        for k in range(K):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def mxu(act: float32[Rt, K], wgt: float32[K, Ct], psum: float32[Rt, Ct]):
        spmw.map(pe, grid=spmw.mesh((Rt, Ct)))
        spmw.stream_in(act, into=pe, flow="W->E")
        spmw.stream_in(wgt, into=pe, flow="N->S")
        spmw.stream_out(psum, from_=pe, where="local", as_="c_local")

    @spmw.unit
    def act_pe(ctx):  # a 1-D vector engine: one lane per output column, bias + ReLU
        j = ctx.rank()
        for r in range(Rt):
            x: float32 = ctx.col_in.get()
            y: float32 = x + ctx.bias[j]
            ctx.col_out.put(y if y > 0.0 else 0.0)

    @spmw.region()
    def mini_tpu(
        ACT: float32[Rt, K],
        WGT: float32[K, Ct],
        bias: float32[Ct],
        OUT: float32[Rt, Ct],
    ):
        # on-chip memory hierarchy: activations + weights are shared buffers, psum is banked per column
        ub = spmw.shared(float32[Rt, K], space="L2")
        wbuf = spmw.shared(float32[K, Ct], space="L2")
        psum = spmw.banked(float32[Rt, Ct], on=psum_bank, space="L2")
        spmw.place(ACT, ub)  # ACT lives in the Unified Buffer
        spmw.place(WGT, wbuf)  # WGT lives in the weight buffer
        mxu(ACT, WGT, psum)  # nested systolic MXU: reads ACT/WGT, writes psum
        # activation: bias + ReLU as a 1-D vector map over the Ct output columns
        spmw.map(act_pe, grid=spmw.Grid((Ct,)))
        spmw.stream_in(psum, into=act_pe, as_="col_in")  # psum column j -> lane j
        spmw.stream_out(OUT, from_=act_pe, as_="col_out")

    return mini_tpu


def _oracle(ACT, WGT, bias):
    return np.maximum(ACT @ WGT + bias, 0.0)


def test_mini_tpu_composes_heterogeneous_stages():
    """L0: the region is a genuine hierarchy -- a nested MXU sub-region + a distinct activation unit,
    with the UB/WGT shared placements and a per-column banked psum."""
    region = _mini_tpu(4, 4, 8)
    collection = spmw._validate_collection(spmw._collect(region))
    # hierarchy by composition: mini_tpu instantiates the mxu sub-region
    assert collection.nested == ["mxu"]
    # two heterogeneous stages: a 2-D systolic MXU and a 1-D vector activation, different units
    dims = sorted(decl.topology.dims for decl in collection.maps)
    assert dims == [1, 2]
    by_dim = {decl.topology.dims: decl.unit.name for decl in collection.maps}
    assert by_dim[2] == "pe" and by_dim[1] == "act_pe"
    # memory hierarchy: ACT/WGT placed in shared buffers at L2 (place() records the operand name);
    # the L2 space level resolves to the URAM resource on the target Memory model
    placed = {p.tensor: p.buffer for p in collection.placements}
    assert placed["ACT"].kind == "shared" and placed["WGT"].kind == "shared"
    assert placed["ACT"].memory.resource == "URAM"
    assert placed["WGT"].memory.resource == "URAM"
    # the psum connecting the two stages is a per-column banked buffer (bank on the col axis == 1)
    psum_buffers = {
        s.tensor
        for s in collection.streams
        if getattr(s.tensor, "kind", None) == "banked"
    }
    assert len(psum_buffers) == 1
    psum = psum_buffers.pop()
    assert psum.bank_axis == 1


def test_mini_tpu_simulator_matches_numpy():
    """L1: the composed Mini-TPU is simulator-correct vs ``np.maximum(ACT @ WGT + bias, 0)``."""
    Rt, Ct, K = 4, 4, 8
    np.random.seed(0)
    ACT = np.random.rand(Rt, K).astype(np.float32)
    WGT = np.random.rand(K, Ct).astype(np.float32)
    bias = np.random.rand(Ct).astype(np.float32)
    OUT = np.zeros((Rt, Ct), dtype=np.float32)
    mod = spmw.build(_mini_tpu(Rt, Ct, K), target="simulator")
    mod(ACT, WGT, bias, OUT)
    np.testing.assert_allclose(OUT, _oracle(ACT, WGT, bias), atol=1e-4)
    # the bias is genuinely applied (all-positive inputs => ReLU is identity here, so the result must
    # differ from the bare matmul): guards against a desugar that silently drops the activation stage
    assert not np.allclose(OUT, ACT @ WGT, atol=1e-4)


def test_mini_tpu_relu_actually_clamps():
    """L1 negative: a large negative bias forces ReLU to clamp, so the activation is not vacuous."""
    Rt, Ct, K = 4, 4, 8
    np.random.seed(1)
    ACT = np.random.rand(Rt, K).astype(np.float32)
    WGT = np.random.rand(K, Ct).astype(np.float32)
    bias = np.full(Ct, -5.0, dtype=np.float32)  # push every pre-activation negative
    OUT = np.zeros((Rt, Ct), dtype=np.float32)
    mod = spmw.build(_mini_tpu(Rt, Ct, K), target="simulator")
    mod(ACT, WGT, bias, OUT)
    pre = ACT @ WGT + bias
    assert (pre < 0).any()  # the test actually exercises the clamp
    np.testing.assert_allclose(OUT, np.maximum(pre, 0.0), atol=1e-4)
    assert np.allclose(OUT[pre < 0], 0.0)  # ReLU clamped the negatives to 0


def test_mini_tpu_requires_col_banked_psum():
    """The psum connection must be ``banked(on="col")`` -- the activation reads one column per lane, so
    per-column banking is what makes that access conflict-free. A row-banked psum fails closed with a
    hard error (the shape is a mini-TPU, so it is a configuration error, not a fall-through).
    """
    with pytest.raises(spmw.SPMWError, match="banked\\(on='col'\\)"):
        spmw.build(_mini_tpu(4, 4, 8, psum_bank="row"), target="simulator")


@pytest.mark.skipif(
    not hls.is_available("vitis_hls"), reason="requires the Vitis HLS toolchain"
)
def test_mini_tpu_vitis_hls_csim():
    """L2: the composed Mini-TPU csims correct on Vitis HLS vs ``np.maximum(ACT @ WGT + bias, 0)``."""
    Rt, Ct, K = 4, 4, 8
    np.random.seed(0)
    ACT = np.random.rand(Rt, K).astype(np.float32)
    WGT = np.random.rand(K, Ct).astype(np.float32)
    bias = np.random.rand(Ct).astype(np.float32)
    OUT = np.zeros((Rt, Ct), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        module = spmw.build(
            _mini_tpu(Rt, Ct, K), target="vitis_hls", mode="csim", project=tmp
        )
        module(ACT, WGT, bias, OUT)
    np.testing.assert_allclose(OUT, _oracle(ACT, WGT, bias), atol=1e-4)


@pytest.mark.skipif(
    not hls.is_available("vitis_hls"), reason="requires the Vitis HLS toolchain"
)
def test_mini_tpu_csyn_emits_synthesizable_project():
    """L3: ``mode="csyn"`` complete-partitions the operands (so HLS dataflow sees a single reader/
    writer per bank) and emits a synthesizable Vitis project. csynth itself runs out of band -- it
    completes on Vitis 2023.2/u280 (Fmax ~411 MHz; see the archived report under examples/).
    """
    with tempfile.TemporaryDirectory() as tmp:
        spmw.build(_mini_tpu(4, 4, 8), target="vitis_hls", mode="csyn", project=tmp)
        assert os.path.exists(os.path.join(tmp, "run.tcl"))
        assert os.path.exists(os.path.join(tmp, "kernel.cpp"))


@pytest.mark.skipif(
    not hls.is_available("vitis_hls"), reason="requires the Vitis HLS toolchain"
)
def test_mini_tpu_hw_emu_emits_emulation_project():
    """L4: ``mode="hw_emu"`` emits a full v++ emulation project (kernel + OpenCL host + Makefile). The
    RTL hardware-emulation run itself is done out of band."""
    with tempfile.TemporaryDirectory() as tmp:
        spmw.build(_mini_tpu(4, 4, 8), target="vitis_hls", mode="hw_emu", project=tmp)
        for artifact in ("kernel.cpp", "host.cpp", "Makefile"):
            assert os.path.exists(os.path.join(tmp, artifact))
