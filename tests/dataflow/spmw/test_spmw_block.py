# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The block engine against Gemmini's arithmetic, one activation at a time.

The reference is `spmw_block_ref`, a transcription of Gemmini's source rather
than a convenience, so a pass here is evidence about the *specification* and
not about two implementations agreeing with each other.

`target="simulator"` is absent on purpose.  The LLVM JIT cannot resolve
`math.sqrt`, which the engine does not use -- the square root is integer, as
Gemmini's is -- but the JIT path was checked and this design does not need it.
The pair that matters is the reference simulator here and the RTL
cosimulation in the array build.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from spmw_block_drive import SPMW_BLOCK_ORDER, launch_operands, MODE_NAME
from spmw_block_engine import block_engine, M_NONE, M_RELU, M_LN, M_GELU, M_SM


def _run(dim, mode, nrow, ln, target="ref", seed=0):
    nacc = nrow * (ln // dim)
    eng = block_engine(dim=dim, tiles=4, nacc=nacc, nrow=nrow)
    # The weight stream differs between the two disciplines, so it is asked
    # for rather than assumed: a file-form stream into a reloading cell reads
    # as an arithmetic bug in every output.
    ops, want, gemm = launch_operands(dim, 4, mode, nrow, ln, seed,
                                      reload_=eng.spmw_shape["reload"])
    spmw.build(eng, target=target)(*[ops[n] for n in SPMW_BLOCK_ORDER])
    return ops, want, gemm


@pytest.mark.parametrize("mode", [M_NONE, M_RELU, M_LN, M_GELU, M_SM])
def test_every_activation_is_bit_exact(mode):
    """All five of Gemmini's activations, on the 16x16 engine."""
    ops, want, _ = _run(16, mode, nrow=4, ln=64)
    np.testing.assert_array_equal(ops["Y"], want, err_msg=MODE_NAME[mode])


@pytest.mark.parametrize("ln", [32, 64, 256])
def test_layernorm_over_the_row_lengths_the_block_uses(ln):
    """`len` is a runtime field, so the row length must not be baked in."""
    ops, want, _ = _run(16, M_LN, nrow=2, ln=ln)
    np.testing.assert_array_equal(ops["Y"], want)


@pytest.mark.parametrize("dim", [4, 8, 16])
def test_the_array_scales(dim):
    """One role per stage, whatever the width."""
    ops, want, _ = _run(dim, M_SM, nrow=2, ln=4 * dim)
    np.testing.assert_array_equal(ops["Y"], want)


def test_the_mesh_still_computes_its_gemm():
    """The mesh half is E3's cell, and the normalise path must not disturb it.

    Both pipelines live in one fabric and share no channel; this is the check
    that they also share no state, which is the claim the resource comparison
    rests on -- Gemmini's mesh and scale path are equally disjoint.
    """
    ops, _, gemm = _run(8, M_LN, nrow=2, ln=32, seed=3)
    np.testing.assert_array_equal(ops["Psum"], gemm)
