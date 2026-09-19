# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The normalise/scale path in NumPy, with Gemmini's exact arithmetic.

This is the third thing both engines are checked against, so that "they agree"
is a fact about a specification and not about two implementations having the
same bug.  Every operation is int32-wrapping or IEEE single where Gemmini's is,
and the rounding modes are the ones its source names:

  * `mean`        `Arithmetic.divider`, a 32-bit-significand float with
                  `round_minMag` throughout -- exactly C integer division
  * `variance`    the *same* divider again: the sum of squared deviations
                  over `count`, truncated, before the square root
  * `stddev`      `IntSqrt`, restoring, exact `floor(sqrt(x))`
  * `inv_stddev`  `Arithmetic.reciprocal`, IEEE single
  * `inv_sum_exp` `127.0f / sum`, IEEE single, the 127 being Gemmini's
  * the scale     `INToRecFN` -> `MulAddRecFN` -> `RecFNToIN`, all
                  `round_near_even`, then a saturating clip to int8
"""

import numpy as np

M_NONE, M_RELU, M_LN, M_GELU, M_SM = 0, 1, 2, 3, 4


def i32(x):
    """int32 wrapping, which is Chisel's `.withWidthOf(q)`."""
    return np.asarray(x).astype(np.int64).astype(np.int32).astype(np.int64)


def _trunc_div(x, n):
    """Integer division truncating towards zero, which is what Gemmini's
    `round_minMag` float round trip amounts to.  NumPy's `//` floors."""
    x = np.asarray(x, dtype=np.int64)
    return np.where(x < 0, -((-x) // n), x // n)


def igelu(q, qb, qc):
    q = i32(q)
    q_sign = np.where(q < 0, -1, 1)
    q_abs = np.where(q < 0, -q, q)
    q_clipped = np.minimum(q_abs, -qb)
    q_poly = i32(i32(q_clipped + qb) * i32(q_clipped + qb) + qc)
    q_erf = i32(q_sign * q_poly)
    return i32(q * i32(q_erf + qc))


def iexp(q, qln2, qln2_inv, qb, qc):
    """`AccumulatorScale.iexp`, including the two truncations to 32 bits."""
    q = i32(q)
    z = i32((-q) * qln2_inv >> 16)          # the product is wide, then cut
    z_sat = np.where(((z >> 5) & 2047) != 0, 32, z)
    qp = i32(q + i32(z * qln2))
    poly = i32(i32(qp + qb) * i32(qp + qb) + qc)
    return np.where(z_sat < 32, poly >> np.minimum(np.maximum(z_sat, 0), 31), 0)


def isqrt32(v):
    """Gemmini's `IntSqrt`, transcribed: restoring, two bits a step."""
    v = np.asarray(v, dtype=np.int64)
    out = np.zeros_like(v)
    for idx in np.ndindex(v.shape):
        x = int(np.uint32(v[idx]))
        a = 0
        q = 0
        for _ in range(16):
            hi = (x >> 30) & 3
            ac = ((a << 2) | hi) & 0xFFFFFFFF
            tt = (ac - (((q << 2) & 0xFFFFFFFF) | 1)) & 0xFFFFFFFF
            neg = 1 if (tt & 0x80000000) else 0
            a = ac if neg else tt
            q = ((q << 1) | (1 - neg)) & 0xFFFF
            x = (x << 2) & 0xFFFFFFFF
        out[idx] = q
    return out


def rne_clip8(fv):
    """`RecFNToIN` with `round_near_even`, then the saturating clip to int8.

    C's cast truncates, so the tie rule is written out.  It matters: the
    requantisation scale is a power of two, which puts an exact half on one
    value in every 256.
    """
    fv = np.clip(np.asarray(fv, dtype=np.float32), -4096.0, 4096.0)
    iv = fv.astype(np.int32).astype(np.int64)
    fr = (fv - iv.astype(np.float32)).astype(np.float32)
    iv = np.where(fr > 0.5, iv + 1, iv)
    iv = np.where(fr < -0.5, iv - 1, iv)
    iv = np.where((fr == 0.5) & ((iv & 1) != 0), iv + 1, iv)
    iv = np.where((fr == -0.5) & ((iv & 1) != 0), iv - 1, iv)
    return np.clip(iv, -128, 127)


def norm_path(acc, mode, scale, qb, qc, qln2, qln2_inv):
    """One launch of the normalise/scale path.

    `acc` is ``[nrow, len]`` -- the accumulator rows the host feeds in, each
    one a full normalisation row.  Returns the int8 result of the same shape,
    and the two scalars per row for cross-checking against the hardware.
    """
    acc = np.asarray(acc, dtype=np.int64)
    nrow, ln = acc.shape
    f32 = np.float32

    if mode == M_LN:
        tot = i32(acc.sum(axis=1))
        mean = _trunc_div(tot, ln).reshape(-1, 1)
        # The **mean** of the squared deviations, not their sum: Gemmini runs
        # its one divider a second time, `get_sum` -> `get_variance`, and the
        # square root sees `sum/count`.  Taking the sum instead makes every
        # output `sqrt(len)` too large -- a clean factor of 8 at `len = 64`,
        # which is what the first run against the RTL showed.
        ssq = i32(i32((acc - mean) ** 2).sum(axis=1))
        var = _trunc_div(ssq, ln)
        sd = np.maximum(isqrt32(var), 1)
        inv = (f32(scale) / sd.astype(f32)).astype(f32).reshape(-1, 1)
        e = acc - mean
    elif mode == M_SM:
        mx = acc.max(axis=1).reshape(-1, 1)
        e = iexp(acc - mx, qln2, qln2_inv, qb, qc)
        tot = i32(e.sum(axis=1)).reshape(-1, 1)
        inv = (f32(scale) * (f32(127.0) / tot.astype(f32))).astype(f32)
        mean = mx
    else:
        if mode == M_GELU:
            e = igelu(acc, qb, qc)
        elif mode == M_RELU:
            e = np.maximum(acc, 0)
        else:
            e = acc
        inv = np.full((nrow, 1), f32(scale), dtype=f32)
        mean = np.zeros((nrow, 1), dtype=np.int64)

    y = rne_clip8(e.astype(f32) * inv)
    return y.astype(np.int8), mean.ravel(), inv.ravel()
