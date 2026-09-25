# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLaMA-3.2-1B's FFN up projections on SPMW's blocked array.

LLaMA's feed-forward block is ``down(SiLU(x W_gate) * (x W_up))``. Its two up
projections share ``x`` and are ``d_model x d_ff`` each -- 2048 x 8192 in
LLaMA-3.2-1B, with no bias -- and over a prompt of ``L`` tokens (prefill)
they are matrix products, which is what an array is for.

Two engines, one stimulus:

* ``gateup`` -- both projections as one GEMM over ``[W_gate | W_up]``,
  requantised to int8 and nothing else::

      G = clip(x W_gate >> s, -128, 127)     U = clip(x W_up >> s, -128, 127)

  Every engine in E9 runs this on chip: VTA as its GEMM and three ALU passes,
  Gemmini as its mesh, accumulator and scale unit.

* ``swiglu`` -- the same sums with SwiGLU fused into the lanes::

      H = clip(isilu(G) * U >> s2, -128, 127)

  which neither Gemmini nor VTA can compute: neither has a SiLU, nor a way
  to multiply two results element by element.

Both are `blocked_engine`, the 256-unit lean cell run on any array size, with
a block of ``M = L`` rows: each weight block is loaded once and every token
streams through it. The simulated slice is all 64 tokens and the full
``K = 2048``, over ``n`` of each projection's 8,192 columns. The operands are
random int8, since no engine's timing depends on values.
"""

import argparse

import numpy as np
import pytest

import allo.spmw as spmw

from test_spmw_tpu_micro import CLIP_HI, CLIP_LO, INT8_MAX
from test_spmw_tpu_micro_blocked import (
    blocked_engine,
    blocked_golden,
    blocked_operands,
)

#: The model: its hidden size is the projections' K, its FFN width their N.
LLAMA_3_2_1B = {"d_model": 2048, "d_ff": 8192}
#: Prompt tokens: prefill, where the projections are matrix products.
TOKENS = 64
#: Columns of each projection simulated, of ``d_ff``.
SLICE = 64


def requant(acc, shift):
    """A bare requantisation: arithmetic shift, clip to int8."""
    return np.clip(np.asarray(acc, dtype=np.int64) >> shift, -128, INT8_MAX)


def signed_shift(acc):
    """The smallest shift whose int8 clip saturates `CLIP_LO` to `CLIP_HI`.

    `test_spmw_tpu_micro.choose_shift` for a result with no ReLU in front:
    derived from the data, so both bounds of the clip are exercised.
    """
    acc = np.asarray(acc, dtype=np.int64)
    for shift in range(32):
        v = acc >> shift
        frac = float(np.mean((v > INT8_MAX) | (v < -128)))
        if frac <= CLIP_HI:
            if frac < CLIP_LO:
                raise ValueError(f"shift {shift} clips only {frac:.4f}")
            return shift
    raise ValueError("the accumulator never exceeds int8")


def isilu(g):
    """SiLU of an int8 ``g`` read as ``g / 16``, in units of ``1 / 4096``.

    I-BERT's clipped second-order polynomial, for a sigmoid:
    ``tanh(x / 2)`` is ``1 - (1 - |x| / 4)^2`` up to ``|x| = 4`` and 1 beyond,
    ``sigmoid(x) = (1 + tanh(x / 2)) / 2`` in 1/256ths, times ``g``. Integer
    throughout, exactly as the swiglu lane computes it.
    """
    g = np.asarray(g, dtype=np.int64)
    w = 64 - np.minimum(np.abs(g), 64)
    th = 256 - ((w * w) >> 4)
    sig = np.where(g < 0, 256 - th, 256 + th) >> 1
    return g * sig


def llama_stimulus(L=TOKENS, K=LLAMA_3_2_1B["d_model"], n=SLICE, seed=0):
    """``x``, the ``[W_gate | W_up]`` slice, both shifts and both goldens."""
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, size=(L, K)).astype(np.int8)
    w = rng.integers(-128, 128, size=(K, 2 * n)).astype(np.int8)
    acc = x.astype(np.int64) @ w.astype(np.int64)
    shift = signed_shift(acc)
    gu = requant(acc, shift)
    prod = isilu(gu[:, :n]) * gu[:, n:]
    shift2 = signed_shift(prod)
    return x, w, shift, shift2, gu.astype(np.int8), requant(prod, shift2)


def llama_of(S, variant="gateup", L=TOKENS, K=LLAMA_3_2_1B["d_model"], n=SLICE):
    """The slice on an ``S x S`` array: ``gateup`` or ``swiglu``."""
    if variant not in ("gateup", "swiglu"):
        raise ValueError(f"variant {variant!r} is not gateup or swiglu")
    epilogue = "requant" if variant == "gateup" else "swiglu"
    x, w, shift, shift2, gu, h = llama_stimulus(L, K, n)
    engine = blocked_engine(S, tiles=1, M=L, K=K, N=2 * n, epilogue=epilogue)
    a, stream, consts = blocked_operands(
        S, x[None], w, None, shift, M=L, epilogue=epilogue, shift2=shift2
    )
    want = blocked_golden(S, (gu if variant == "gateup" else h)[None])
    engine.spmw_operands = {"A": a, "W": stream, "Bias": consts}
    # Checked against numpy: the reference simulator is checked against the
    # same golden below, at sizes it can run.
    engine.spmw_expected = {"Y": want}
    engine.spmw_tokens_per_transform = L * S  # one column block's outputs
    engine.spmw_cosim_cycles = len(a) + 64 * (L + S) + 10000
    engine.spmw_llama = {
        "size": S,
        "variant": variant,
        "lkn": (L, K, n),
        "shift": int(shift),
        "shift2": int(shift2),
        "expected": want,
    }
    return engine


def dump_llama(L=TOKENS, K=LLAMA_3_2_1B["d_model"], n=SLICE, seed=0):
    """The shared stimulus as whitespace-separated integers.

    `test_spmw_tpu_micro.dump`'s format with the tile's shape spelled out:
    ``A`` is ``x``, ``B`` is ``[W_gate | W_up]``, shared by every row, ``C`` is
    the gate-and-up result and ``H`` the SwiGLU one. No bias line: LLaMA's
    projections have none.
    """
    x, w, shift, shift2, gu, h = llama_stimulus(L, K, n, seed)
    flat = lambda a: " ".join(str(int(v)) for v in np.asarray(a).reshape(-1))
    return "\n".join(
        [
            f"M {L}",
            f"K {K}",
            f"N {2 * n}",
            "TILES 1",
            f"SHIFT {shift}",
            f"SHIFT2 {shift2}",
            f"SEED {seed}",
            "A " + flat(x),
            "B " + flat(w),
            "C " + flat(gu),
            "H " + flat(h),
            "",
        ]
    )


# -- tests --------------------------------------------------------------------


def test_isilu_tracks_silu():
    """Within 0.1 of SiLU over int8's whole range, read as ``g / 16``.

    The worst point is 0.090, at ``x = -3.7``, where the polynomial has
    reached 0 and SiLU is still -0.09; the RMS error is 0.034, about half of
    the input's own step of 1/16.
    """
    g = np.arange(-128, 128)
    x = g / 16.0
    err = isilu(g) / 4096.0 - x / (1.0 + np.exp(-x))
    assert np.abs(err).max() < 0.1, np.abs(err).max()
    assert np.sqrt(np.mean(err**2)) < 0.04


def test_the_slice_is_llama_3_2_1b():
    x, w, shift, shift2, gu, h = llama_stimulus(L=16, n=16)
    assert x.shape == (16, 2048) and w.shape == (2048, 32)
    assert gu.shape == (16, 32) and h.shape == (16, 16)
    assert 0 < shift < 32 and 0 < shift2 < 32


@pytest.mark.parametrize("variant", ["gateup", "swiglu"])
@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_the_engines_match_the_golden(variant, target):
    """A small slice -- 16 tokens, K = 32, eight columns a projection."""
    engine = llama_of(4, variant, L=16, K=32, n=8)
    want = engine.spmw_llama["expected"]
    Y = np.zeros(want.shape, dtype=np.int32)
    ops = [engine.spmw_operands[k] for k in ("A", "W", "Bias")]
    spmw.build(engine, target=target)(*ops, Y)
    np.testing.assert_array_equal(Y, want)


def test_swiglu_combines_what_gateup_emits():
    """The fused lane's inputs are exactly the three-way engine's outputs."""
    x, w, shift, shift2, gu, h = llama_stimulus(L=16, K=32, n=8)
    np.testing.assert_array_equal(
        h, requant(isilu(gu[:, :8]) * gu[:, 8:].astype(np.int64), shift2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="write the shared stimulus out")
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument("--k", type=int, default=LLAMA_3_2_1B["d_model"])
    parser.add_argument("--n", type=int, default=SLICE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(dump_llama(args.tokens, args.k, args.n, args.seed))
    print(f"wrote {args.out}")
