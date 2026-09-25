# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""E3's microbenchmark at a fixed size, on an array of any size.

`test_spmw_tpu_micro_lean` sizes the tile to the array, M = K = N = S, so a
4x4 array and a 16x16 one run different workloads.  Here the workload is fixed
-- T tiles of M x K x N, sixteen of 16 x 16 x 16 by default -- and the array
shrinks, which is what a scaling study needs:

    Y_t[m, n] = min(127, max(0, sum_k A_t[m, k] B_t[k, n] + b[n]) >> shift)

An S x S array takes each tile as ``K/S x N/S`` weight blocks.  Block
``(nb, kb)`` holds ``B_t[kb*S + i, nb*S + j]`` in cell ``(i, j)`` and is fed
the ``M`` rows of ``A_t[:, kb*S:(kb+1)*S]``; the ``K/S`` partial sums of each
output are added up in the lanes before the epilogue.  So the ideal is
``T*M*K*N / S^2`` cycles: 256, 1,024 and 4,096 at S = 16, 8 and 4.

The cell is the 256-unit lean cell unchanged -- Gemmini's weight shift, one
multiply-add between bare-register links, one loop -- except that a block
now lasts ``M`` rows rather than ``S``: the weight chain shifts ``M`` tokens a
block, and the stream puts the block's ``S`` weights in the last ``S`` of
them.  What is new is the lane: an ``M``-deep delay line holds each row's
running sum from one ``kb`` pass to the next, so the lane adds, and emits only
on the last pass.  At ``S = K`` there is one pass and no delay line.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int8, int32

from test_spmw_tpu_micro import INT8_MAX, TILES, stimulus_mkn
from test_spmw_tpu_micro_lean import _generated

M_ROWS = 16  # rows a weight block is fed: the tile's M


def _log2(n):
    bits = n.bit_length() - 1
    if 1 << bits != n:
        raise ValueError(f"{n} is not a power of two")
    return bits


#: What a lane does once a row's ``K`` blocks are summed. ``relu`` is E3's
#: epilogue -- bias, ReLU, shift, clip to 127. ``requant`` is a bare
#: requantisation, shift and a clip to int8 on both sides, for a layer with no
#: bias and no activation. ``swiglu`` fuses LLaMA's SwiGLU: the lane sums a
#: gate block and an up block for the same outputs and emits
#: ``SiLU(gate) * up`` (see `test_spmw_llama_ffn`).
EPILOGUES = ("relu", "requant", "swiglu")


def _lane_source(rows, kb, nb, epilogue="relu"):
    """The lane: a running sum over ``kb`` passes, then the epilogue.

    ``r0 .. r{rows-1}`` is the delay line: the value pushed ``rows`` steps ago
    is the same row's sum from the previous pass, so ``r0`` is that row's head.
    """
    if epilogue == "requant":
        return _requant_lane_source(rows, kb)
    if epilogue == "swiglu":
        return _swiglu_lane_source(rows, kb)
    delay = kb > 1
    lines = [
        "def make(total, LM, LKB, NB, KB, INT8_MAX, IO):",
        "    def lane(io: IO):",
        "        sh: int32 = io.b[NB] & 31",
    ]
    if delay:
        lines += [f"        r{i}: int32 = 0" for i in range(rows)]
    lines += [
        "        for s in range(total):",
        "            kb: int32 = (s >> LM) & (KB - 1)",
        "            nb: int32 = (s >> (LM + LKB)) & (NB - 1)",
        "            z: int32 = io.z_in.get()",
        "            base: int32 = io.b[nb]",
    ]
    if delay:
        lines += [
            "            if kb != 0:",
            "                base = r0",
        ]
    lines.append("            v: int32 = base + z")
    if delay:
        lines += [f"            r{i} = r{i + 1}" for i in range(rows - 1)]
        lines.append(f"            r{rows - 1} = v")
    lines += [
        "            if kb == KB - 1:",
        "                if v < 0:",
        "                    v = 0",
        "                v = v >> sh",
        "                if v > INT8_MAX:",
        "                    v = INT8_MAX",
        "                io.y_out.put(v)",
        "    return lane",
    ]
    return "\n".join(lines) + "\n"


def _clip8(name, indent):
    """Lines clipping ``name`` to int8, both sides."""
    pad = " " * indent
    return [
        f"{pad}if {name} < -128:",
        f"{pad}    {name} = -128",
        f"{pad}if {name} > INT8_MAX:",
        f"{pad}    {name} = INT8_MAX",
    ]


def _shift_line(depth):
    """Lines pushing ``v`` into the delay line ``r0 .. r{depth-1}``."""
    pad = " " * 12
    return [f"{pad}r{i} = r{i + 1}" for i in range(depth - 1)] + [
        f"{pad}r{depth - 1} = v"
    ]


def _requant_lane_source(rows, kb):
    """A running sum, then a bare requantisation: shift, clip to int8.

    No bias and no activation: ``b`` is the shift alone. A pass's first block
    starts from zero, so nothing is preloaded.
    """
    delay = kb > 1
    lines = [
        "def make(total, LM, KB, INT8_MAX, IO):",
        "    def lane(io: IO):",
        "        sh: int32 = io.b[0] & 31",
    ]
    if delay:
        lines += [f"        r{i}: int32 = 0" for i in range(rows)]
    lines += [
        "        for s in range(total):",
        "            kb: int32 = (s >> LM) & (KB - 1)",
        "            z: int32 = io.z_in.get()",
        "            v: int32 = z",
    ]
    if delay:
        lines += ["            if kb != 0:", "                v = r0 + z"]
        lines += _shift_line(rows)
    lines += ["            if kb == KB - 1:", "                v = v >> sh"]
    lines += _clip8("v", 16)
    lines += ["                io.y_out.put(v)", "    return lane"]
    return "\n".join(lines) + "\n"


def _swiglu_lane_source(rows, kb):
    """Gate and up sums side by side, then ``SiLU(gate) * up``, requantised.

    The blocks come gate, up, gate, up: for each ``kb`` a gate block and then
    the up block for the same columns. So the delay line is ``2 * rows`` deep
    -- its head ``r0`` is the same row and projection one ``kb`` ago -- and
    during the last up block the finished gate sum for the same row sits at
    ``r{rows}``. Both are requantised to int8 with ``b[0]``, which is exactly
    what the gate-and-up GEMM emits, then combined with the integer SiLU of
    `test_spmw_llama_ffn.isilu` and requantised again with ``b[1]``.

    The SiLU is a clipped second-order polynomial in int8, I-BERT's form:
    ``tanh(x/2)`` for ``x = g/16`` is ``1 - (1 - |x|/4)^2`` up to ``|x| = 4``
    and 1 beyond. Its three multiplies are 8 x 8, 8 x 16 and 16 x 8 bits.
    """
    depth = 2 * rows
    lines = [
        "def make(total, LM, KB, INT8_MAX, IO):",
        "    def lane(io: IO):",
        "        sh: int32 = io.b[0] & 31",
        "        sh2: int32 = io.b[1] & 31",
    ]
    lines += [f"        r{i}: int32 = 0" for i in range(depth)]
    lines += [
        "        for s in range(total):",
        "            blk: int32 = s >> LM",
        "            p: int32 = blk & 1",
        "            kb: int32 = (blk >> 1) & (KB - 1)",
        "            z: int32 = io.z_in.get()",
        "            v: int32 = z",
    ]
    if kb > 1:
        lines += ["            if kb != 0:", "                v = r0 + z"]
    lines.append(f"            g: int32 = r{rows}")
    lines += _shift_line(depth)
    lines += [
        "            if kb == KB - 1:",
        "                if p == 1:",
        "                    gq: int32 = g >> sh",
    ]
    lines += _clip8("gq", 20)
    lines.append("                    uq: int32 = v >> sh")
    lines += _clip8("uq", 20)
    lines += [
        "                    a: int32 = gq",
        "                    if a < 0:",
        "                        a = -a",
        "                    if a > 64:",
        "                        a = 64",
        "                    w: int8 = 64 - a",
        "                    th: int16 = 256 - ((w * w) >> 4)",
        "                    sig: int16 = 256 + th",
        "                    if gq < 0:",
        "                        sig = 256 - th",
        "                    sig = sig >> 1",
        "                    g8: int8 = gq",
        "                    u8: int8 = uq",
        "                    gs: int16 = g8 * sig",
        "                    h: int32 = gs * u8",
        "                    h = h >> sh2",
    ]
    lines += _clip8("h", 20)
    lines += ["                    io.y_out.put(h)", "    return lane"]
    return "\n".join(lines) + "\n"


def block_order(tiles, KB, NB, epilogue="relu"):
    """The weight blocks in stream order, as ``(t, kb, column block)``.

    Tile by tile, then column block, then ``kb``. For ``swiglu`` the ``N``
    columns are the gate's and then the up projection's, and each ``kb``
    streams the gate block and then the up block for the same columns.
    """
    if epilogue != "swiglu":
        return [
            (t, kb, nb) for t in range(tiles) for nb in range(NB) for kb in range(KB)
        ]
    half = NB // 2
    return [
        (t, kb, p * half + nb)
        for t in range(tiles)
        for nb in range(half)
        for kb in range(KB)
        for p in range(2)
    ]


def blocked_engine(S, tiles=TILES, M=M_ROWS, K=16, N=16, epilogue="relu"):
    """An ``S x S`` array of lean cells running ``tiles`` tiles of M x K x N.

    ``epilogue`` is one of `EPILOGUES`. For ``swiglu`` the ``N`` columns are
    the gate's and the up projection's, and the result has ``N/2``.
    """
    if epilogue not in EPILOGUES:
        raise ValueError(f"epilogue {epilogue!r} is not one of {EPILOGUES}")
    if K % S or N % S or M < S:
        raise ValueError(f"S={S} must divide K={K} and N={N} and not exceed M={M}")
    KB, NB = K // S, N // S
    if epilogue == "swiglu" and NB % 2:
        raise ValueError(f"gate and up need N/S={NB} to be even")
    blocks = tiles * NB * KB
    steps = blocks * M
    last = M - 1
    _log2(M), _log2(KB), _log2(NB)
    out_nb = NB // 2 if epilogue == "swiglu" else NB
    consts = {"relu": NB + 1, "requant": 1, "swiglu": 2}[epilogue]
    # `spmw.pipeline`'s credit rule: E3's lane and the requantising one are
    # two stages, reads then writes, and take one credit; SwiGLU's has stages
    # between them, and a credit there packs its multiplies into one stage.
    lane_credit = 0 if epilogue == "swiglu" else 1

    class PEIO(spmw.Interface):
        """The lean cell's links, every one a bare register."""

        a_in = spmw.In(int8, depth=0)
        a_out = spmw.Out(int8, depth=0)
        p_in = spmw.In(int32, depth=0)
        p_out = spmw.Out(int32, depth=0)
        w_in = spmw.In(int8, depth=0)
        w_out = spmw.Out(int8, depth=0)

    class LaneIO(spmw.Interface):
        z_in = spmw.In(int32, depth=2)
        y_out = spmw.Out(int32)
        # E3's: a bias per column block, then the shift. Otherwise the shifts.
        b = spmw.MemIn(int32[consts])

    mxu = spmw.Topology(
        PEIO,
        grid=(S, S),
        link=lambda i, j: {
            PEIO.a_out: spmw.to((i, j + 1), PEIO.a_in),
            PEIO.w_out: spmw.to((i, j + 1), PEIO.w_in),
            PEIO.p_out: spmw.to((i + 1, j), PEIO.p_in),
        },
    )
    lanes = spmw.Grid((S,))

    @spmw.unit
    def pe(io: PEIO):
        # `lean_engine`'s one-loop cell with a block of `M` rows: the first `M`
        # iterations only shift, and every `M`-th boundary hands the next
        # block's weight to `cur`.
        nxt: int8 = 0
        cur: int8 = 0
        for s in range(steps + M):
            if s >= M:
                a = io.a_in.get()
                p = io.p_in.get()
                io.a_out.put(a)
                io.p_out.put(p + a * cur)
            io.w_out.put(nxt)
            nxt = io.w_in.get()
            if (s & last) == last:
                cur = nxt

    closure = dict(total=steps, LM=_log2(M), KB=KB, INT8_MAX=INT8_MAX, IO=LaneIO)
    if epilogue == "relu":
        closure.update(LKB=_log2(KB), NB=NB)
    tag = {"relu": "", "requant": "q", "swiglu": "sw"}[epilogue]
    lane = _generated(
        f"lane{tag}_k{KB}n{NB}", _lane_source(M, KB, NB, epilogue), **closure
    )

    @spmw.fabric
    def engine(
        A: int8[steps, S],
        W: int8[steps + M, S],
        Bias: int32[S, consts],
        Y: int32[tiles * out_nb * M, S],
    ):
        P = spmw.place(pe, on=mxu)
        spmw.pipeline(P, ii=1, combinational=True)
        V = spmw.place(lane, on=lanes)
        spmw.pipeline(V, ii=1, registered_links=lane_credit == 1)
        spmw.shard(Bias, into=V.b)
        spmw.stream_in(A, into=P.a_in, index=(..., P.rows))
        spmw.stream_in(W, into=P.w_in, index=(..., P.rows))
        spmw.stream_in(0, into=P.p_in)
        spmw.link(P.p_out, to=V.z_in)
        (col,) = V.axes
        spmw.gather(Y, from_=V.y_out, index=(..., col))

    engine.spmw_bind_mul_fabric = True
    return engine


def blocked_operands(S, A, B, bias, shift, M=M_ROWS, epilogue="relu", shift2=0):
    """The array's three input streams, and where each output lands.

    Blocks go in `block_order`; each is fed its ``M`` rows.  The weight stream
    carries a block's weights in the last ``S`` of its ``M`` tokens, columns
    ``S-1 .. 0``, so that after ``M`` shifts cell ``j`` holds column ``j``; one
    block of zeros closes it.  ``B`` is ``[tiles, K, N]``, or ``[K, N]`` when
    every tile shares one weight matrix, as a layer's tokens do.  The lanes'
    constants are E3's biases and shift for ``relu``, the shift for
    ``requant``, and the shift and ``shift2`` for ``swiglu``.
    """
    tiles, _, K = A.shape
    N = B.shape[-1]
    KB, NB = K // S, N // S
    order = block_order(tiles, KB, NB, epilogue)
    a = np.zeros((len(order) * M, S), dtype=np.int8)
    w = np.zeros(((len(order) + 1) * M, S), dtype=np.int8)
    for q, (t, kb, nb) in enumerate(order):
        a[q * M : (q + 1) * M, :] = A[t][:, kb * S : (kb + 1) * S]
        weights = B[t] if B.ndim == 3 else B
        blk = weights[kb * S : (kb + 1) * S, nb * S : (nb + 1) * S]  # [i, j]
        w[q * M + M - S : (q + 1) * M, :] = blk[:, ::-1].T
    if epilogue == "requant":
        return a, w, np.full((S, 1), shift, dtype=np.int32)
    if epilogue == "swiglu":
        return a, w, np.tile(np.array([[shift, shift2]], dtype=np.int32), (S, 1))
    consts = np.zeros((S, NB + 1), dtype=np.int32)
    for j in range(S):
        consts[j, :NB] = bias[np.arange(NB) * S + j]
        consts[j, NB] = shift
    return a, w, consts


def blocked_golden(S, want):
    """The golden ``[tiles, M, N]`` result in the order the lanes emit it."""
    tiles, M, N = want.shape
    NB = N // S
    out = np.zeros((tiles * NB * M, S), dtype=np.int32)
    for t in range(tiles):
        for nb in range(NB):
            rows = slice((t * NB + nb) * M, (t * NB + nb + 1) * M)
            out[rows, :] = want[t][:, nb * S : (nb + 1) * S]
    return out


def micro_blocked_of(S, tiles=TILES, M=M_ROWS, K=16, N=16, seed=0):
    """The fixed workload on an ``S x S`` array, with its stimulus attached."""
    A, B, bias, shift, want = stimulus_mkn(tiles, M, K, N, seed)
    engine = blocked_engine(S, tiles, M, K, N)
    a, w, consts = blocked_operands(S, A, B, bias, shift, M)
    engine.spmw_operands = {"A": a, "W": w, "Bias": consts}
    engine.spmw_tokens_per_transform = (N // S) * M * S
    engine.spmw_micro = {
        "size": S,
        "tiles": tiles,
        "mkn": (M, K, N),
        "shift": int(shift),
        "expected": blocked_golden(S, want),
    }
    return engine


# -- tests --------------------------------------------------------------------


@pytest.mark.parametrize("S", [4, 8, 16])
def test_the_fixed_workload_matches_the_golden_on_every_array(S):
    """Sixteen 16x16x16 tiles on 4x4, 8x8 and 16x16: one golden result."""
    engine = micro_blocked_of(S)
    want = engine.spmw_micro["expected"]
    Y = np.zeros(want.shape, dtype=np.int32)
    ops = [engine.spmw_operands[n] for n in ("A", "W", "Bias")]
    spmw.build(engine, target="ref")(*ops, Y)
    np.testing.assert_array_equal(Y, want)


@pytest.mark.parametrize("S", [4, 8])
def test_blocked_lanes_in_the_simulator(S):
    """The delay-line lane, compiled: two tiles, so it stays quick."""
    engine = micro_blocked_of(S, tiles=2)
    want = engine.spmw_micro["expected"]
    Y = np.zeros(want.shape, dtype=np.int32)
    ops = [engine.spmw_operands[n] for n in ("A", "W", "Bias")]
    spmw.build(engine, target="simulator")(*ops, Y)
    np.testing.assert_array_equal(Y, want)


def test_the_16x16_array_runs_the_microbenchmark_it_always_ran():
    """At S = 16 the blocked workload is E3's: same operands, same result."""
    from test_spmw_tpu_micro_lean import micro_lean_of

    blocked, lean = micro_blocked_of(16), micro_lean_of(16)
    np.testing.assert_array_equal(
        blocked.spmw_micro["expected"], lean.spmw_micro["expected"]
    )
    np.testing.assert_array_equal(blocked.spmw_operands["A"], lean.spmw_operands["A"])
    np.testing.assert_array_equal(blocked.spmw_operands["W"], lean.spmw_operands["W"])


def test_a_transformer_shaped_tile_blocks_the_same_way():
    """M x K x N need not be square: a 16 x 32 x 64 tile on a 4x4 array."""
    engine = micro_blocked_of(4, tiles=1, M=16, K=32, N=64)
    want = engine.spmw_micro["expected"]
    Y = np.zeros(want.shape, dtype=np.int32)
    ops = [engine.spmw_operands[n] for n in ("A", "W", "Bias")]
    spmw.build(engine, target="ref")(*ops, Y)
    np.testing.assert_array_equal(Y, want)
