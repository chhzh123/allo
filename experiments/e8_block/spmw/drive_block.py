# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive the SPMW block engine and check it against `block_ref`."""

import sys

import numpy as np

import allo.spmw as spmw
from block_engine import (
    block_engine, NK, NF, K_MODE, K_NBEAT, K_TOTAL, K_QB, K_QC, K_QLN2,
    K_QLN2I, K_NLEN, M_NONE, M_RELU, M_LN, M_GELU, M_SM,
)
from block_ref import norm_path

#: The quantisation, and the constants that follow from it.  See `block_spec`.
S_ACT = 0.03
IGELU_QB, IGELU_QC = -59, -3847
IEXP_QB, IEXP_QC, IEXP_QLN2, IEXP_QLN2I = 45, 1066, 23, 2849

#: The requantisation scale per activation.  `igelu`'s is negative and is not
#: a free choice: I-BERT's output scale is `a*S^2/2` and `qc = 1/(a*S^2)`, so
#: the scale that puts `igelu`'s integer output back on the activation's own
#: scale is exactly `1/(2*qc)`, sign included.
SCALE = {
    M_NONE: 1.0 / 256,
    M_RELU: 1.0 / 256,
    M_LN: 32.0,          # (x-mean)/stddev is standardised; +-4 sigma fills int8
    M_GELU: 1.0 / (2 * IGELU_QC),
    M_SM: 1.0,           # `127/sum` already spans int8
}


def consts(mode, nbeat, total, ln, qb, qc):
    k = np.zeros(NK, dtype=np.int32)
    k[K_MODE], k[K_NBEAT], k[K_TOTAL] = mode, nbeat, total
    k[K_QB], k[K_QC] = qb, qc
    k[K_QLN2], k[K_QLN2I] = IEXP_QLN2, IEXP_QLN2I
    k[K_NLEN] = ln
    return k


def operands(dim, tiles, acc, mode, seed=0):
    """Every tensor a launch needs, for an `acc` of shape ``[nrow, len]``."""
    nrow, ln = acc.shape
    assert ln % dim == 0
    nbeat, nacc = ln // dim, nrow * (ln // dim)
    outs, kw = tiles * dim, tiles // 4
    rng = np.random.default_rng(seed)
    qb, qc = (IGELU_QB, IGELU_QC) if mode == M_GELU else (IEXP_QB, IEXP_QC)

    # The mesh half runs a plain GEMM alongside; it shares no state with the
    # normalise path, exactly as Gemmini's mesh and scale path do not.
    A = rng.integers(-8, 8, size=(outs, dim), dtype=np.int64).astype(np.int8)
    W = np.zeros((dim * kw + 1, dim), dtype=np.int32)
    W[0, :] = dim * kw
    W[1:, :] = rng.integers(-2**20, 2**20, size=(dim * kw, dim))

    # The accumulator rows, laid out beat by beat: beat b of row r carries
    # columns [b*dim, (b+1)*dim).
    Acc = acc.reshape(nrow, nbeat, dim).reshape(nacc, dim).astype(np.int32)
    return dict(
        A=A, W=W,
        Pin=np.zeros((outs, dim), dtype=np.int32),
        Psum=np.zeros((outs, dim), dtype=np.int32),
        Acc=Acc,
        Kr1=np.tile(consts(mode, nbeat, nacc, ln, qb, qc), (dim, 1)),
        Kr2=np.tile(consts(mode, nbeat, nacc, ln, qb, qc), (dim, 1)),
        Kst1=consts(mode, nbeat, nrow, ln, qb, qc).reshape(1, NK),
        Kst2=consts(mode, nbeat, nrow, ln, qb, qc).reshape(1, NK),
        Fst2=np.array([[np.float32(SCALE[mode]), np.float32(0)]], dtype=np.float32),
        Kscl=np.tile(consts(mode, nbeat, nacc, ln, qb, qc), (dim, 1)),
        Z1=np.zeros(nacc, dtype=np.int32),
        Z2=np.zeros(nacc, dtype=np.int32),
        Tail=np.zeros(nrow, dtype=np.int32),
        Vail=np.zeros(nrow, dtype=np.float32),
        Y=np.zeros((nacc, dim), dtype=np.int8),
    )


ORDER = ("A", "W", "Pin", "Psum", "Acc", "Kr1", "Kr2", "Kst1", "Kst2", "Fst2",
         "Kscl", "Z1", "Z2", "Tail", "Vail", "Y")


def run(dim=16, tiles=4, nrow=4, ln=64, mode=M_LN, target="ref", seed=0):
    rng = np.random.default_rng(seed + 7)
    acc = rng.integers(-3000, 3000, size=(nrow, ln)).astype(np.int64)
    nacc = nrow * (ln // dim)
    eng = block_engine(dim=dim, tiles=tiles, nacc=nacc, nrow=nrow)
    t = operands(dim, tiles, acc, mode, seed)
    spmw.build(eng, target=target)(*[t[n] for n in ORDER])

    qb, qc = (IGELU_QB, IGELU_QC) if mode == M_GELU else (IEXP_QB, IEXP_QC)
    want, mean, inv = norm_path(acc, mode, SCALE[mode], qb, qc,
                                IEXP_QLN2, IEXP_QLN2I)
    got = t["Y"].reshape(nrow, ln // dim, dim).reshape(nrow, ln)
    ok = np.array_equal(got, want)
    name = {M_NONE: "none", M_RELU: "relu", M_LN: "layernorm",
            M_GELU: "igelu", M_SM: "softmax"}[mode]
    print(f"BLOCK {target} {name:9s} dim={dim} nrow={nrow} len={ln} "
          f"{'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        bad = np.argwhere(got != want)[:6]
        for r, c in bad:
            print(f"   [{r},{c}] got {got[r, c]} want {want[r, c]} acc {acc[r, c]}")
        print(f"   mismatching {len(np.argwhere(got != want))} of {got.size}")
        print(f"   tail mean {t['Tail'].ravel()[:4]} want {mean[:4]}")
        print(f"   tail inv  {t['Vail'].ravel()[:4]} want {inv[:4]}")
    return ok


if __name__ == "__main__":
    tgt = sys.argv[1] if len(sys.argv) > 1 else "ref"
    modes = {"none": M_NONE, "relu": M_RELU, "ln": M_LN, "gelu": M_GELU,
             "sm": M_SM}
    which = sys.argv[2:] or ["ln", "sm", "gelu", "none", "relu"]
    allok = True
    for w in which:
        allok &= run(mode=modes[w], target=tgt)
    print("BLOCK_ALL", "OK" if allok else "FAILED")
