#!/usr/bin/env python3
"""The transformer block both engines run, as exact integer arithmetic.

This is the specification, not a convenience: every operation below is the one
Gemmini performs, transcribed from `AccumulatorScale.scala` and
`Normalizer.scala` at commit 8c3f992, so that "the two designs agree" can be
checked against a third thing rather than against each other.

Three things about Gemmini's shipped code that this file has to take a position
on, all verified in its source and all recorded here rather than in a comment
nobody reads:

1. **`iexp` is a copy of `igelu`.** Its live body ignores `qln2`/`qln2_inv` and
   applies the erf polynomial; the real exponential sits above it commented
   out, at `master` and at the pinned commit alike. Shipped, SOFTMAX is not a
   softmax. `scripts/restore_iexp.py` uncomments Gemmini's own exponential and
   this model implements that, because matching the broken one bit-for-bit
   would make the workload not a transformer.

2. **The scale form has to be `mul`.** With `SCALE_MODE=shift`,
   `scale_func(v, sc) = v >> sc(4,0)` reads the scalar as a *shift count*, so
   `inv_stddev` and `inv_sum_exp` would be truncated to five bits. The
   multiply form, `(v * sc) >> 8`, is the one that can carry a reciprocal.

3. **The per-row scalars come from a divide and a square root**, and in Gemmini
   both are *floating point* (`INToRecFN` + hardfloat, `round_minMag`) with a
   `TODO` saying they should be integer. They are **shared** -- one divider,
   one sqrt, one reciprocal for the whole accumulator -- which is what keeps
   them affordable, and is the structure the SPMW side has to mirror if its
   resource numbers are to mean anything.
"""

import numpy as np

# -- the block's shape. Small enough to simulate cycle-accurately on both
# -- engines, large enough to be a real block: multi-head attention with
# -- softmax, two LayerNorms, a GELU feed-forward, and both residuals.
SEQ = 64
D_MODEL = 256
HEADS = 4
D_HEAD = D_MODEL // HEADS
D_FFN = 4 * D_MODEL

#: The quantisation scale, and the polynomial constants that follow from it.
#: Both activations use I-BERT's second-order fits, but **different** ones, and
#: Gemmini reuses the same two CSR fields (`igelu_qb`, `igelu_qc`) for each --
#: its SOFTMAX path literally passes `igelu_qb, igelu_qc` into `iexp` -- so the
#: driver reprograms them per activation. Getting that wrong is silent: the erf
#: constants in `iexp` give a function that goes negative.
#:
#: Calibrated against float references rather than asserted:
#:   igelu vs 0.5x(1+erf(x/sqrt2)) : max absolute error 0.085 over int8
#:   iexp  vs exp                  : exp(-ln2)=0.5015, exp(-2ln2)=0.2507,
#:                                   1.6% max relative error over four binades
S_ACT = 0.03

#: erf: erf(x) ~ sign(x) [a (clip(|x|,0,-b) + b)^2 + 1], a=-0.2888 b=-1.769
IGELU_QB = -59
IGELU_QC = -3847

#: exp: exp(x) ~ a(x+b)^2 + c on [-ln2, 0], a=0.3585 b=1.353 c=0.344
IEXP_QB = 45
IEXP_QC = 1066
IEXP_QLN2 = 23
IEXP_QLN2_INV = 2849


def igelu(q, qb=IGELU_QB, qc=IGELU_QC):
    """`AccumulatorScale.igelu`, transcribed.

        q_sign    = q < 0 ? -1 : 1
        q_abs     = |q|
        q_clipped = min(q_abs, -qb)
        q_poly    = (q_clipped + qb)^2 + qc        # `qc.mac(a, b)` is a*b + qc
        out       = q * (q_sign * q_poly + qc)
    """
    q = np.asarray(q, dtype=np.int64)
    q_sign = np.where(q < 0, -1, 1)
    q_abs = np.abs(q)
    q_clipped = np.minimum(q_abs, -qb)
    q_poly = (q_clipped + qb) ** 2 + qc
    return q * (q_sign * q_poly + qc)


def iexp(q, qln2=IEXP_QLN2, qln2_inv=IEXP_QLN2_INV, qb=IEXP_QB, qc=IEXP_QC):
    """`AccumulatorScale.iexp` **as restored** -- the exponential, not the erf.

        z      = (-q * qln2_inv) >> 16          # q <= 0, so z >= 0
        z_sat  = any bit 5..15 of z ? 32 : z
        qp     = q + z * qln2
        poly   = (qp + qb)^2 + qc                  # `qc.mac(a, b)` is a*b + qc
        out    = poly >> z_sat

    The decomposition is `exp(q) = 2^-z * exp(qp)` with `qp` in one binade, so
    the polynomial only has to be accurate on a bounded interval.
    """
    q = np.asarray(q, dtype=np.int64)
    z = (-q * qln2_inv) >> 16
    z_sat = np.where((z >> 5) != 0, 32, z)
    qp = q + z * qln2
    poly = (qp + qb) ** 2 + qc
    return poly >> np.minimum(z_sat, 63)


def clip8(x):
    return np.clip(x, -128, 127).astype(np.int8)


# -- the per-row statistics, which Gemmini computes in a *shared* unit ---------
#
# One divider, one square root, one reciprocal for the whole accumulator -- not
# one per element. That is what makes them affordable, and it is the structure
# the SPMW side has to mirror for its resource numbers to mean anything.


def layernorm_stats(row):
    """`mean` and `inv_stddev` for one row, as the Normalizer's state machine
    produces them: sum, then sum of squares, then a divide and a sqrt."""
    n = row.shape[-1]
    mean = row.sum(axis=-1, keepdims=True) // n
    var = ((row - mean) ** 2).sum(axis=-1, keepdims=True) // n
    inv_stddev = np.where(var > 0, (1 << 16) // np.maximum(np.sqrt(var).astype(np.int64), 1), 1 << 16)
    return mean, inv_stddev


def softmax_stats(row, qln2=IEXP_QLN2, qln2_inv=IEXP_QLN2_INV, qb=IEXP_QB, qc=IEXP_QC):
    """`max` and `inv_sum_exp`: the maximum, then the exponentials, then a
    reciprocal of their sum. Two passes over the row, as on both engines."""
    mx = row.max(axis=-1, keepdims=True)
    ex = iexp(row - mx, qln2, qln2_inv, qb, qc)
    tot = ex.sum(axis=-1, keepdims=True)
    return mx, ex, np.where(tot > 0, (1 << 16) // np.maximum(tot, 1), 0)


def block(x, w, shift=8):
    """One transformer block, every operation on device.

        h  = x + attn(layernorm(x))
        y  = h + ffn(layernorm(h))

    with softmax inside the attention, GELU inside the feed-forward, and both
    LayerNorms and both residuals. Returns the int8 output and a trace of every
    GEMM's shape so the cycle model can be built from the same description the
    functional check uses.
    """
    trace = []

    def gemm(a, b):
        trace.append((a.shape[0], a.shape[1], b.shape[1]))
        return a.astype(np.int64) @ b.astype(np.int64)

    def ln(t):
        mean, inv = layernorm_stats(t)
        return clip8(((t - mean) * inv) >> 16)

    def sm(t):
        _, ex, inv = softmax_stats(t)
        return clip8((ex * inv) >> 16)

    a = ln(x)
    qkv = gemm(a, w["qkv"])
    q, k, v = (qkv[:, i * D_MODEL:(i + 1) * D_MODEL] for i in range(3))
    heads = []
    for h in range(HEADS):
        sl = slice(h * D_HEAD, (h + 1) * D_HEAD)
        qh, kh, vh = clip8(q[:, sl] >> shift), clip8(k[:, sl] >> shift), clip8(v[:, sl] >> shift)
        heads.append(gemm(sm(gemm(qh, kh.T)), vh))
    ctx = clip8(np.concatenate(heads, axis=1) >> shift)
    h1 = clip8(x + clip8(gemm(ctx, w["proj"]) >> shift))

    b = ln(h1)
    f1 = clip8(igelu(gemm(b, w["ffn1"]) >> shift) >> shift)
    return clip8(h1 + clip8(gemm(f1, w["ffn2"]) >> shift)), trace
