# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operands for one launch of the block engine, and the golden it must match.

Kept apart from the engine so that the array build, the reference check and
the cosimulation all hand the hardware the *same* tensors.  When the golden
and the stimulus are built in two places they drift, and the drift shows up as
a hardware bug.
"""

import numpy as np

from spmw_block_engine import (
    NK, NF, K_MODE, K_NBEAT, K_TOTAL, K_QB, K_QC, K_QLN2, K_QLN2I, K_NLEN,
    M_NONE, M_RELU, M_LN, M_GELU, M_SM,
)
from spmw_block_ref import norm_path

#: The quantisation.  `S_ACT` is the activation scale the I-BERT constants are
#: fitted at; the two polynomials use **different** constants and Gemmini
#: reuses the same two CSR fields for both, so the driver reprograms them.
S_ACT = 0.03
IGELU_QB, IGELU_QC = -59, -3847
IEXP_QB, IEXP_QC, IEXP_QLN2, IEXP_QLN2I = 45, 1066, 23, 2849

#: The requantisation scale per activation.
#:
#: `igelu`'s is **negative**, and it is derived rather than tuned: I-BERT's
#: output scale is `a*S^2/2` and `qc = round(1/(a*S^2))`, so the scale that
#: puts `igelu`'s integer output back on the activation's own scale is exactly
#: `1/(2*qc)`.  A least-squares fit against the true GELU over the whole int8
#: range gives -1.2878e-4 against the derived -1.2997e-4, within the
#: polynomial's own 2.2% error, so the derivation is what is used.
#:
#: LayerNorm's 32 is a choice: `(x-mean)/stddev` is standardised, so 32 puts
#: four standard deviations at the edge of int8.  Softmax's 1.0 is not a
#: choice -- `inv_sum_exp` already carries Gemmini's own 127.
SCALE = {
    M_NONE: 1.0 / 256,
    M_RELU: 1.0 / 256,
    M_LN: 32.0,
    M_GELU: 1.0 / (2 * IGELU_QC),
    M_SM: 1.0,
}

MODE_NAME = {M_NONE: "none", M_RELU: "relu", M_LN: "layernorm",
             M_GELU: "igelu", M_SM: "softmax"}
NAME_MODE = {v: k for k, v in MODE_NAME.items()}

SPMW_BLOCK_ORDER = ("A", "W", "Pin", "Psum", "Acc1", "Acc2", "Acc3",
                    "Kr1", "Kr2", "Ksu1", "Ksc1", "Ksu2", "Ksc2", "Fsc2",
                    "Kscl", "Z1", "Z2",
                    "Tail", "Vail", "Y")


def site_consts(mode, nbeat, total, ln, qb, qc):
    k = np.zeros(NK, dtype=np.int32)
    k[K_MODE], k[K_NBEAT], k[K_TOTAL] = mode, nbeat, total
    k[K_QB], k[K_QC] = qb, qc
    k[K_QLN2], k[K_QLN2I] = IEXP_QLN2, IEXP_QLN2I
    k[K_NLEN] = ln
    return k


def launch_operands(dim, tiles, mode, nrow, ln, seed=0):
    """Every tensor one launch needs, the int8 result, and the mesh's GEMM."""
    if ln % dim:
        raise ValueError(f"a row of {ln} does not divide into {dim} lanes")
    nbeat, nacc = ln // dim, nrow * (ln // dim)
    outs, kw = tiles * dim, tiles // 4
    rng = np.random.default_rng(seed + 7)
    qb, qc = (IGELU_QB, IGELU_QC) if mode == M_GELU else (IEXP_QB, IEXP_QC)

    # Accumulator rows at the magnitude a real psum reaches: a 256-deep int8
    # dot product is a few thousand, which is also where `iexp`'s wide
    # intermediate and the variance's 32-bit wrap both start to matter.
    acc = rng.integers(-3000, 3000, size=(nrow, ln)).astype(np.int64)

    # The mesh half runs a GEMM alongside and shares no state with the
    # normalise path, exactly as Gemmini's mesh and scale path do not.
    A = rng.integers(-8, 8, size=(outs, dim)).astype(np.int8)
    wt8 = rng.integers(-128, 128, size=(tiles, dim, dim)).astype(np.int64)
    # The packed weight file, four int8 to a 32-bit word: the word cell
    # `(row k, column j)` holds sits at `W[1 + j*kw + t//4][k]`, because the
    # load is serial down the row and cell `j` takes the `kw` words that reach
    # it after the `j` cells before it have taken theirs.
    W = np.zeros((dim * kw + 1, dim), dtype=np.int64)
    W[0, :] = dim * kw
    for t in range(tiles):
        for j in range(dim):
            W[1 + j * kw + t // 4, :] |= (wt8[t, :, j] & 255) << ((t & 3) * 8)
    W = W.astype(np.int32)

    k_lane = site_consts(mode, nbeat, nacc, ln, qb, qc)
    k_stat = site_consts(mode, nbeat, nrow, ln, qb, qc)
    ops = dict(
        A=A, W=W,
        Pin=np.zeros((outs, dim), dtype=np.int32),
        Psum=np.zeros((outs, dim), dtype=np.int32),
        Acc1=acc.reshape(nacc, dim).astype(np.int32),
        Acc2=acc.reshape(nacc, dim).astype(np.int32),
        Acc3=acc.reshape(nacc, dim).astype(np.int32),
        Kr1=np.tile(k_lane, (dim, 1)),
        Kr2=np.tile(k_lane, (dim, 1)),
        Ksu1=k_lane.reshape(1, NK),
        Ksc1=k_stat.reshape(1, NK),
        Ksu2=k_lane.reshape(1, NK),
        Ksc2=k_stat.reshape(1, NK),
        Fsc2=np.array([[np.float32(SCALE[mode]), np.float32(0)]],
                      dtype=np.float32),
        Kscl=np.tile(k_lane, (dim, 1)),
        Z1=np.zeros(nacc, dtype=np.int32),
        Z2=np.zeros(nacc, dtype=np.int32),
        Tail=np.zeros(nrow, dtype=np.int32),
        Vail=np.zeros(nrow, dtype=np.float32),
        Y=np.zeros((nacc, dim), dtype=np.int8),
    )
    gemm = np.concatenate(
        [A[t * dim:(t + 1) * dim].astype(np.int64) @ wt8[t]
         for t in range(tiles)], axis=0).astype(np.int32)
    want, _, _ = norm_path(acc, mode, SCALE[mode], qb, qc, IEXP_QLN2,
                           IEXP_QLN2I)
    return ops, want.reshape(nacc, dim), gemm
