# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Transformer block, run on the TPU by handing it instructions.

`test_spmw_tpu_isa.py` builds the engine: a matrix unit whose instruction picks
which weight matrix to use, and a vector unit with a register file and a program
of its own. Nothing in it knows about attention. This file writes the
instructions that make it compute one.

**One fabric runs the whole block.** Every step below is the same silicon --
the same exported IPs, the same netlist -- handed a different weight tensor, a
different MXU program and a different VPU program. That is the claim being
tested, and the reason each step is an engine *invocation* rather than a new
design.

    step  what                              MXU program            VPU program
    ----  --------------------------------  ---------------------  ------------
     1-3  Q, K, V = X.Wq, X.Wk, X.Wv        MACC tile 0 / 1 / 2    requantise
      4   S' = K.Q'                         MACC tile 0            requantise
      5   row max of S'                     MACC identity          running MAX
      6   row sum of exp(S' - max)          MACC identity          EXP2, sum
      7   P' = exp(S' - max) / sum          MACC identity          EXP2, LOADR
      8   O = P.V                           MACC tile 0            requantise
      9   Y = O.Wo + X                      MACC 0 / MACC identity two ACCZ
     10   H = relu(Y.W1)                    MACC tile 0            MAX 0, shift
     11   Out = H.W2 + Y                    MACC 0 / MACC identity two ACCZ

Three things in that table are worth pausing on.

**Steps 1-3 share one weight load.** Wq, Wk and Wv sit in tiles 0, 1 and 2 of
the same weight tensor, and only the instruction differs. That is what the MXU's
tile field bought.

**Steps 5-7 do no matrix arithmetic at all.** They select the *identity* tile, so
the array passes its activations through untouched and the pass is a pure vector
operation. A VPU-only pass needs no separate datapath -- it is an instruction.

**Steps 9 and 11 add the residual in the MXU.** `Y = O.Wo + X` is two matmuls,
one against Wo and one against the identity, interleaved a row at a time and
folded back together by two ACCZ instructions in the lane. The skip connection
is not special-cased anywhere; it is an extra MACC.

What is *not* on the engine, stated plainly: the host requantises int32
accumulators back to int8 between steps, transposes P, and feeds each step's
reductions back as the next step's per-lane constants. A real TPU does the first
in its accumulator path and the second in a transpose unit; both are memory
formatting rather than arithmetic. The arithmetic is all here.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from test_spmw_tpu_isa import (
    ACCZ,
    ADD,
    D,
    EXP2,
    LOADB,
    LOADI,
    LOADZ,
    MACC,
    MAX,
    MSKIP,
    MUL,
    LOADR,
    MZERO,
    NOP,
    REGS,
    NB,
    NPROG,
    NW,
    RCP_BITS,
    SEQ,
    SHR,
    STORE,
    SUB,
    _engine,
    mxu_program,
    vpu_program,
)

# The array feeds two accumulators per output, so one fabric serves both the
# plain matmuls (the second is a skipped bubble) and the residual adds (the
# second is the skip connection).
STEPS = 2 * SEQ
OUTS = SEQ
IDENT = NW - 1  # the weight tile holding the identity matrix
PER_OUT = STEPS // OUTS  # accumulators the array feeds per output

# Fixed-point scales, one per step rather than one for the block -- which is
# what a quantised Transformer does, and it is not optional. A single shift for
# everything collapsed the block to -1 and 0: a matmul of random signs grows by
# about sqrt(D)*|w| while a shift of 4 divides by 16, so the magnitude halved at
# every step and the feed-forward ReLU finished it off. Each shift below is
# picked from the range its own step actually produces.
QUANT_QKV = 4  # X.W with X ~ +/-64, W ~ +/-8
QUANT_SCORE = 6  # K.Q' -- both operands are already full-range
QUANT_OUT = 4  # the projection and the feed-forward layers
EXP_SHIFT = 5  # how much of a score's range one exp step covers
EXP_BASE = 8  # exp(0) is 1 << EXP_BASE
PROB_BITS = 6  # fractional bits in an attention weight: 1.0 is 64
RECIP_BITS = RCP_BITS  # the lane's prologue divide, not an immediate


class Shape:
    """One size of the engine: the array's side and the sequence it processes.

    Everything below takes a `Shape`, so the same block runs on a 4x4 array for
    the tests and on a 16x16 one for the board. The programs do not change with
    it -- an MXU program is broadcast across whatever rows exist, and a VPU
    program is per-lane -- which is the point.
    """

    def __init__(self, dim, seq):
        self.dim = dim
        self.seq = seq
        self.steps = 2 * seq
        self.outs = seq
        self.per_out = 2
        self.engine = _engine(self.steps, outs=self.outs, dim=dim)


#: The size the tests run at.
SMALL = Shape(D, SEQ)
#: The size taken to the board: a 16x16 matrix unit and a 16-lane vector unit,
#: so a model dimension of 16 and a sequence of 16. Same roles, same programs,
#: 272 instances instead of 20.
BIG = Shape(16, 16)
engine = SMALL.engine


# -- the programs -----------------------------------------------------------


def matmul_prog(tile, shape=None):
    """A plain matmul: use the tile on even steps, skip the odd ones."""
    shape = shape or SMALL
    return mxu_program(
        [(MACC, tile), (MSKIP,)] * shape.outs, rows=shape.dim, pad=shape.steps
    )


def residual_prog(tile, shape=None):
    """A matmul and an identity pass, interleaved; the lane sums the pair."""
    shape = shape or SMALL
    return mxu_program(
        [(MACC, tile), (MACC, IDENT)] * shape.outs, rows=shape.dim, pad=shape.steps
    )


def requant(shift, outs=None):
    """Sum the pair of accumulators and rescale."""
    return vpu_program(
        [(LOADI, 0, 0, 0), (ACCZ, 0), (ACCZ, 0), (SHR, 0, 0, shift), (STORE, 0)],
        reads=PER_OUT,
        outs=outs or SMALL.outs,
    )


def relu_requant(shift, outs=None):
    """The same, clamped at zero first."""
    return vpu_program(
        [
            (LOADI, 0, 0, 0),
            (ACCZ, 0),
            (ACCZ, 0),
            (LOADI, 1, 0, 0),
            (MAX, 0, 1),
            (SHR, 0, 0, shift),
            (STORE, 0),
        ],
        reads=PER_OUT,
        outs=outs or SMALL.outs,
    )


# A running maximum over the pass. `reg` persists across steps, and r0 starts at
# zero, so this is max(0, scores) -- which is still a legal softmax shift,
# because subtracting any constant leaves softmax unchanged.
def row_max(outs=None):
    """A running maximum over the pass."""
    return vpu_program(
        [(LOADZ, 1), (ACCZ, 1), (MAX, 0, 1), (STORE, 0)],
        reads=PER_OUT,
        outs=outs or SMALL.outs,
    )


# exp(s - max), as a shift. `1 << (EXP_BASE + (s - max) >> EXP_SHIFT)`, clamped
# at both ends by EXP2 itself.
_EXP = [
    (LOADZ, 1),
    (ACCZ, 1),  # r1 = this step's score (two accumulators, one skipped)
    (LOADB, 2, 0),  # r2 = the row maximum
    (SUB, 1, 2),
    (SHR, 1, 0, EXP_SHIFT),
    (LOADI, 2, 0, EXP_BASE),
    (ADD, 1, 2),
    (EXP2, 1),
]


def row_sum(outs=None):
    """exp(s - max), accumulated."""
    return vpu_program(
        _EXP + [(ADD, 0, 1), (STORE, 0)], reads=PER_OUT, outs=outs or SMALL.outs
    )


def normalise(outs=None):
    """exp(s - max) / sum, in PROB_BITS fixed point."""
    return vpu_program(
        _EXP
        + [
            # The lane divided by the row sum once, before the pass; this just
            # reads the result. The divide used to be here, and cost 17x.
            (LOADR, 3),
            (MUL, 1, 3),
            # Not all the way back: a weight lands in [0, 1<<PROB_BITS],
            # because shifting the whole way would truncate every probability
            # to zero.
            (SHR, 1, 0, RECIP_BITS - PROB_BITS),
            (STORE, 1),
        ],
        reads=PER_OUT,
        outs=outs or SMALL.outs,
    )


# -- the host's side of the loop --------------------------------------------


# What the build script must load to bring this engine up: a legal pair of
# programs. Random small integers decode to opcodes that do not exist, and an
# MXU program of those would feed the lane the wrong number of accumulators.
engine.spmw_operands = {
    "MProg": mxu_program([(MACC, 0), (MSKIP,)] * SEQ, pad=SMALL.steps),
    "VProg": vpu_program(
        [(LOADI, 0, 0, 0), (ACCZ, 0), (ACCZ, 0), (SHR, 0, 0, QUANT_QKV), (STORE, 0)],
        reads=2,
        outs=SMALL.outs,
    ),
}


def _interleave(rows, other=None, shape=None):
    """Lay activations out for the array: one row per step.

    A plain matmul pairs each row with a bubble the program skips; a residual
    pairs it with the row the skip connection carries.
    """
    shape = shape or SMALL
    out = np.zeros((shape.steps, shape.dim), dtype=np.int8)
    out[0::2] = rows
    if other is not None:
        out[1::2] = other
    return out


def _weights(*mats, shape=None):
    """Pack up to NW-1 matrices into the cell weight file, identity last."""
    shape = shape or SMALL
    w = np.zeros((shape.dim, shape.dim, NW), dtype=np.int8)
    for tile, mat in enumerate(mats):
        w[:, :, tile] = mat
    w[:, :, IDENT] = np.eye(shape.dim, dtype=np.int8)
    return w


def _clip8(x):
    """What the accumulator path does on the way back to an int8 activation."""
    return np.clip(x, -128, 127).astype(np.int8)


class Engine:
    """One built fabric, invoked repeatedly. The only thing that changes is data.

    Every invocation is recorded, because the interesting question about this
    design is not whether one pass works -- it is whether the *same* netlist
    computes all eleven. `scripts/spmw_transformer_rtl.py` replays the trace
    against the exported IPs, one xsim run per step.
    """

    def __init__(self, target="ref", label=None, shape=None):
        self.shape = shape or SMALL
        self.run = spmw.build(self.shape.engine, target=target)
        self.trace = []
        self.label = label or []

    @property
    def invocations(self):
        return len(self.trace)

    def __call__(self, acts, weights, vprog, mprog, bias=None, name=""):
        dim, outs = self.shape.dim, self.shape.outs
        bias = np.zeros((dim, NB), dtype=np.int32) if bias is None else bias
        out = np.zeros((outs, dim), dtype=np.int32)
        self.run(acts, weights, bias.astype(np.int32), mprog, vprog, out)
        self.trace.append(
            {
                "name": name or f"step{len(self.trace) + 1}",
                "A": acts.copy(),
                "W": weights.copy(),
                "Bias": bias.astype(np.int32).copy(),
                "MProg": mprog.copy(),
                "VProg": vprog.copy(),
                "Y": out.copy(),
            }
        )
        return out


def transformer_block(x, wq, wk, wv, wo, w1, w2, target="ref", shape=None):
    """One Transformer block, as a sequence of instruction-driven passes.

    Returns the block's output and the engine, so a test can ask how many times
    the same fabric was invoked.
    """
    shape = shape or SMALL
    eng = Engine(target=target, shape=shape)
    ident = _weights(shape=shape)

    def w_(*mats):
        return _weights(*mats, shape=shape)

    def lay(rows, other=None):
        return _interleave(rows, other, shape=shape)

    def mprog(tile):
        return matmul_prog(tile, shape=shape)

    def rprog(tile):
        return residual_prog(tile, shape=shape)

    qkv = w_(wq, wk, wv)

    # 1-3. Three projections, one weight load, three instructions.
    q, k, v = (
        _clip8(
            eng(
                lay(x),
                qkv,
                requant(QUANT_QKV, shape.outs),
                mprog(tile),
                name=f"proj{tile}",
            )
        )
        for tile in range(3)
    )

    # 4. S' = K . Q', so a lane holds one query's scores over every key and the
    #    softmax that follows is a reduction *along* the lane rather than across.
    scores = _clip8(
        eng(lay(k), w_(q.T), requant(QUANT_SCORE, shape.outs), mprog(0), name="scores")
    )

    # 5-7. Softmax, three passes over the same scores through the identity tile.
    maxes = eng(lay(scores), ident, row_max(shape.outs), mprog(IDENT), name="row_max")[
        -1
    ]
    bias = np.zeros((shape.dim, NB), dtype=np.int32)
    bias[:, 0] = maxes
    sums = eng(
        lay(scores), ident, row_sum(shape.outs), mprog(IDENT), bias, name="row_sum"
    )[-1]
    bias[:, 1] = sums
    probs_t = eng(
        lay(scores), ident, normalise(shape.outs), mprog(IDENT), bias, name="softmax"
    )

    # 8. O = P . V. The lane held S transposed, so the host hands back P.
    probs = _clip8(probs_t.T)
    attn = _clip8(
        eng(lay(probs), w_(v), requant(PROB_BITS, shape.outs), mprog(0), name="attn")
    )

    # 9. Y = O . Wo + X -- the residual is an extra MACC against the identity.
    y = _clip8(
        eng(
            lay(attn, x),
            w_(wo),
            requant(QUANT_OUT, shape.outs),
            rprog(0),
            name="proj_out",
        )
    )

    # 10-11. The feed-forward network, and its residual.
    h = _clip8(
        eng(lay(y), w_(w1), relu_requant(QUANT_OUT, shape.outs), mprog(0), name="ffn1")
    )
    out = eng(lay(h, y), w_(w2), requant(QUANT_OUT, shape.outs), rprog(0), name="ffn2")
    return out, eng


# -- a reference for the same arithmetic ------------------------------------


def _ref_block(x, wq, wk, wv, wo, w1, w2):
    """The same computation in numpy, step for step and shift for shift.

    Not a float Transformer: this is what the *integer* block means, so a
    mismatch is a hardware bug rather than a quantisation effect. How far this
    sits from a float block is a separate question, and
    `test_the_integer_softmax_is_a_softmax` is where it is asked.
    """
    i32 = np.int32

    def mm(a, b, shift):
        return (a.astype(i32) @ b.astype(i32)) >> shift

    q, k, v = (_clip8(mm(x, w, QUANT_QKV)) for w in (wq, wk, wv))
    scores = _clip8(mm(k, q.T, QUANT_SCORE))  # scores[key, query]
    row_max = np.maximum(scores.max(axis=0), 0).astype(i32)
    arg = np.clip(EXP_BASE + ((scores.astype(i32) - row_max) >> EXP_SHIFT), 0, 30)
    exps = (np.int32(1) << arg).astype(i32)
    row_sum = exps.sum(axis=0)
    recip = np.where(
        row_sum > 0, (np.int32(1) << RECIP_BITS) // np.maximum(row_sum, 1), 0
    )
    probs_t = (exps * recip) >> (RECIP_BITS - PROB_BITS)
    probs = _clip8(probs_t.T)
    attn = _clip8((probs.astype(i32) @ v.astype(i32)) >> PROB_BITS)
    y = _clip8((attn.astype(i32) @ wo.astype(i32) + x.astype(i32)) >> QUANT_OUT)
    h = _clip8(np.maximum(y.astype(i32) @ w1.astype(i32), 0) >> QUANT_OUT)
    return (h.astype(i32) @ w2.astype(i32) + y.astype(i32)) >> QUANT_OUT


def _params(seed=5, shape=None):
    """Activations near the full int8 range, weights small -- as quantised."""
    shape = shape or SMALL
    rng = np.random.default_rng(seed)

    def weight():
        return rng.integers(-8, 8, (shape.dim, shape.dim)).astype(np.int8)

    x = rng.integers(-64, 64, (shape.seq, shape.dim)).astype(np.int8)
    return x, weight(), weight(), weight(), weight(), weight(), weight()


# -- tests ------------------------------------------------------------------


def test_the_reference_is_not_degenerate():
    """A block whose output is mostly zeros would not test much."""
    out = _ref_block(*_params())
    assert np.count_nonzero(out) >= out.size // 2, out
    assert len(np.unique(out)) >= 6, np.unique(out)


def test_the_integer_softmax_is_a_softmax():
    """The shift-based exp is coarse, but it is still a distribution.

    Every weight non-negative, every row summing to about one in the chosen
    fixed point, and the largest score getting the largest weight. That last one
    is the property attention actually depends on.
    """
    x, wq, wk, wv, *_ = _params()
    i32 = np.int32
    kk = _clip8((x.astype(i32) @ wk.astype(i32)) >> QUANT_QKV)
    qq = _clip8((x.astype(i32) @ wq.astype(i32)) >> QUANT_QKV)
    scores = _clip8((kk.astype(i32) @ qq.T.astype(i32)) >> QUANT_SCORE)
    row_max = np.maximum(scores.max(axis=0), 0).astype(i32)
    arg = np.clip(EXP_BASE + ((scores.astype(i32) - row_max) >> EXP_SHIFT), 0, 30)
    exps = (np.int32(1) << arg).astype(i32)
    probs_t = (exps * ((np.int32(1) << RECIP_BITS) // exps.sum(axis=0))) >> (
        RECIP_BITS - PROB_BITS
    )
    one = 1 << PROB_BITS
    assert (probs_t >= 0).all()
    for lane in range(D):
        total = probs_t[:, lane].sum()
        assert 0.9 * one <= total <= 1.1 * one, (lane, total, one)
        assert probs_t[scores[:, lane].argmax(), lane] == probs_t[:, lane].max()


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_the_block_runs_on_the_engine(target):
    """The whole block, computed by invoking one fabric eleven times."""
    params = _params()
    out, eng = transformer_block(*params, target=target)
    np.testing.assert_array_equal(out, _ref_block(*params))
    assert eng.invocations == 11


def test_it_is_one_fabric_throughout():
    """Eleven steps, one design: nothing here builds a second engine."""
    params = _params()
    _out, eng = transformer_block(*params)
    assert eng.invocations == 11
    graph = spmw.elaborate(engine)
    assert len(graph.placements) == 2  # the MXU and the VPU chain, and that is all


def test_the_projections_share_one_weight_load():
    """Q, K and V differ only by an instruction, which is the MXU ISA's point."""
    x, wq, wk, wv, *_ = _params()
    eng = Engine()
    qkv = _weights(wq, wk, wv)
    got = [
        eng(_interleave(x), qkv, requant(QUANT_QKV), matmul_prog(t)) for t in range(3)
    ]
    for tile, weight in enumerate((wq, wk, wv)):
        want = (x.astype(np.int32) @ weight.astype(np.int32)) >> QUANT_QKV
        np.testing.assert_array_equal(got[tile], want)
    assert not np.array_equal(got[0], got[1])


def test_the_residual_is_an_extra_macc():
    """`Y = O.Wo + X` with no adder outside the array."""
    x, wq, *_ = _params()
    eng = Engine()
    o = _clip8(x)
    got = eng(_interleave(o, x), _weights(wq), requant(QUANT_OUT), residual_prog(0))
    want = (o.astype(np.int32) @ wq.astype(np.int32) + x.astype(np.int32)) >> QUANT_OUT
    np.testing.assert_array_equal(got, want)


def test_a_vpu_only_pass_selects_the_identity():
    """Steps 5-7 do no matrix arithmetic; the array is a wire."""
    x, *_ = _params()
    eng = Engine()
    passthrough = vpu_program(
        [(LOADI, 0, 0, 0), (ACCZ, 0), (ACCZ, 0), (STORE, 0)],
        reads=PER_OUT,
        outs=OUTS,
    )
    got = eng(_interleave(x), _weights(), passthrough, matmul_prog(IDENT))
    np.testing.assert_array_equal(got, x.astype(np.int32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Programs the design was not written for
# ---------------------------------------------------------------------------
#
# The eleven steps above show that a Transformer needs no rebuild, which is a
# weaker claim than it looks: they are the programs this engine was written for.
# These are not. One of them issues MZERO, an MXU opcode the block never uses.
# `scripts/spmw_program_check.py` runs every one of them against the netlist the
# Transformer was simulated on, without rebuilding anything.

_NOVEL_X = np.arange(-64, -64 + STEPS * D, dtype=np.int64).reshape(STEPS, D)


def _novel_inputs(seed=11):
    rng = np.random.default_rng(seed)
    acts = rng.integers(-40, 40, (STEPS, D)).astype(np.int8)
    w = np.zeros((D, D, NW), dtype=np.int8)
    for tile in range(NW - 1):
        w[:, :, tile] = rng.integers(-6, 6, (D, D)).astype(np.int8)
    w[:, :, IDENT] = np.eye(D, dtype=np.int8)
    return acts, w


def _pairs(acts, w, tile):
    """The two accumulators the array feeds each output, as int64."""
    prod = acts.astype(np.int64) @ w[:, :, tile].astype(np.int64)
    return prod[0::2], prod[1::2]


def _square(acts, w):
    lo, hi = _pairs(acts, w, 0)
    return ((lo + hi) * (lo + hi)) >> 8


def _maxpair(acts, w):
    lo, hi = _pairs(acts, w, 0)
    return np.maximum(lo, hi)


def _absval(acts, w):
    lo, hi = _pairs(acts, w, 0)
    return np.abs(lo + hi)


def _mzero(acts, w):
    """MZERO restarts the column at every row, so only the last row survives."""
    last = D - 1
    return np.outer(acts[0::2, last].astype(np.int64), w[last, :, 1].astype(np.int64))


def _loadr(acts, w, bias1):
    """LOADR yields the same per-lane constant at every element."""
    lo, hi = _pairs(acts, w, 0)
    rcp = np.where(bias1 > 0, (np.int64(1) << RCP_BITS) // np.maximum(bias1, 1), 0)
    return np.broadcast_to(rcp, lo.shape).copy()


#: ``(name, MXU program, VPU program, what it should compute)``.
NOVEL_PROGRAMS = [
    (
        "square",
        mxu_program([(MACC, 0), (MACC, 0)] * OUTS, pad=STEPS),
        vpu_program(
            [
                (LOADI, 0, 0, 0),
                (ACCZ, 0),
                (ACCZ, 0),
                (MUL, 0, 0),
                (SHR, 0, 0, 8),
                (STORE, 0),
            ],
            reads=PER_OUT,
            outs=OUTS,
        ),
        _square,
    ),
    (
        "max_of_pair",
        mxu_program([(MACC, 0), (MACC, 0)] * OUTS, pad=STEPS),
        vpu_program(
            [(LOADZ, 0), (LOADZ, 1), (MAX, 0, 1), (STORE, 0)],
            reads=PER_OUT,
            outs=OUTS,
        ),
        _maxpair,
    ),
    (
        "abs",
        mxu_program([(MACC, 0), (MACC, 0)] * OUTS, pad=STEPS),
        vpu_program(
            [
                (LOADI, 0, 0, 0),
                (ACCZ, 0),
                (ACCZ, 0),
                (LOADI, 1, 0, 0),
                (SUB, 1, 0),
                (MAX, 0, 1),
                (STORE, 0),
            ],
            reads=PER_OUT,
            outs=OUTS,
        ),
        _absval,
    ),
    (
        "mzero",  # an MXU opcode the Transformer never issues
        mxu_program([(MZERO, 1), (MSKIP,)] * OUTS, pad=STEPS),
        vpu_program(
            [(LOADI, 0, 0, 0), (ACCZ, 0), (ACCZ, 0), (STORE, 0)],
            reads=PER_OUT,
            outs=OUTS,
        ),
        _mzero,
    ),
    (
        "loadr",  # the pass reciprocal, read as an operand
        mxu_program([(MACC, 0), (MACC, 0)] * OUTS, pad=STEPS),
        vpu_program(
            [(LOADI, 0, 0, 0), (ACCZ, 0), (ACCZ, 0), (LOADR, 0), (STORE, 0)],
            reads=PER_OUT,
            outs=OUTS,
        ),
        _loadr,
    ),
]


def novel_operands(name):
    """The tensors one novel program is driven with, and what it should give."""
    acts, w = _novel_inputs()
    entry = next(e for e in NOVEL_PROGRAMS if e[0] == name)
    _n, mprog, vprog, expect = entry
    bias = np.zeros((D, NB), dtype=np.int32)
    # A lane's prologue divides by b[1], so a program that reads the result
    # needs a divisor worth dividing by.
    bias[:, 1] = np.arange(3, 3 + D, dtype=np.int32)
    want = expect(acts, w, bias[:, 1]) if name == "loadr" else expect(acts, w)
    return {
        "A": acts,
        "W": w,
        "Bias": bias,
        "MProg": mprog,
        "VProg": vprog,
        "Y": want.astype(np.int32),
    }


@pytest.mark.parametrize("name", [e[0] for e in NOVEL_PROGRAMS])
def test_a_novel_program_runs_on_the_same_design(name):
    """Each computes what it says, on the engine built for something else."""
    arrays = novel_operands(name)
    eng = Engine()
    got = eng(
        arrays["A"], arrays["W"], arrays["VProg"], arrays["MProg"], arrays["Bias"]
    )
    np.testing.assert_array_equal(got, arrays["Y"])


def test_the_novel_programs_are_novel():
    """None of them is one of the block's, and one uses an opcode it never does."""
    _out, eng = transformer_block(*_params())
    used = {step["VProg"].tobytes() for step in eng.trace}
    for name, _m, vprog, _f in NOVEL_PROGRAMS:
        assert vprog.tobytes() not in used, name
    block_opcodes = {
        (int(w) >> 24) & 0xFF for step in eng.trace for w in step["MProg"].reshape(-1)
    }
    assert MZERO not in block_opcodes, "the block already uses MZERO"
    novel = {
        (int(w) >> 24) & 0xFF for _n, m, _v, _f in NOVEL_PROGRAMS for w in m.reshape(-1)
    }
    assert MZERO in novel


def test_the_shape_is_data_and_the_buffers_are_hardware():
    """What a rebuild costs, and what it does not.

    A TPU is fixed silicon that takes shapes as instructions, and this design
    used to fail that: `steps`, `outs` and the program length were loop bounds
    in the units, so a different shape meant a different netlist. They are now
    read off the head of the instruction stream -- the MXU is told how many
    instructions follow, the VPU how long its program is and how many results
    to emit.

    What stays hardware is what is physically there: the array's size, the
    register file, the instruction buffer, the cell weight file. That is the
    same split a real machine makes.
    """
    import re

    from allo.spmw.role_ip import UnitEmitter, build_unit, trim_includes

    graph = spmw.elaborate(engine)
    emitter = UnitEmitter(graph)
    code = "".join(
        trim_includes(str(build_unit(graph, placement, 0, target="vhls").hls_code))
        for placement in emitter.placements()
    )
    bounds = re.findall(r"for \(int (\w+) = 0; \1 < ([^;]+);", code)
    assert bounds, "no loops found; the check would pass vacuously"

    # Every loop with a *literal* bound is walking a physical buffer, and its
    # bound is that buffer's size. Nothing else is compiled in.
    literals = sorted({int(b) for _v, b in bounds if b.strip().isdigit()})
    assert literals == sorted({NW, NB, NPROG, REGS}), literals

    # And the loops that carry the shape -- the MXU's step count, the VPU's
    # program length and output count -- are bounded by values read at runtime.
    assert sum(1 for _v, b in bounds if not b.strip().isdigit()) >= 4

    # A program that does not fit the buffer is still refused, in Python.
    with pytest.raises(ValueError, match="buffer holds"):
        vpu_program([(NOP,)] * (NPROG + 1))


def test_one_engine_runs_two_program_lengths():
    """The same elaborated design, two different programs, no rebuild.

    A five-instruction requantise and a nine-instruction softmax normalise run
    on one engine. Before the header they could not: the lane's loop bound was
    the buffer size, so every program had to be exactly that long.
    """
    a, w = _novel_inputs()
    bias = np.zeros((D, NB), dtype=np.int32)
    bias[:, 1] = 64
    eng = Engine()

    short = vpu_program(
        [(LOADI, 0, 0, 0), (ACCZ, 0), (ACCZ, 0), (SHR, 0, 0, 2), (STORE, 0)],
        reads=PER_OUT,
        outs=OUTS,
    )
    long_ = vpu_program(
        [
            (LOADI, 0, 0, 0),
            (ACCZ, 0),
            (ACCZ, 0),
            (LOADR, 1),
            (MUL, 0, 1),
            (SHR, 0, 0, 8),
            (LOADI, 2, 0, 0),
            (MAX, 0, 2),
            (STORE, 0),
        ],
        reads=PER_OUT,
        outs=OUTS,
    )
    assert (int(short[0]) & 0xFFFF) == 5 and (int(long_[0]) & 0xFFFF) == 9

    prog = mxu_program([(MACC, 0), (MACC, 0)] * OUTS, pad=STEPS)
    lo, hi = _pairs(a, w, 0)
    got_s = eng(a, w, short, prog, bias)
    np.testing.assert_array_equal(got_s, ((lo + hi) >> 2).astype(np.int32))

    rcp = (np.int64(1) << RCP_BITS) // 64
    got_l = eng(a, w, long_, prog, bias)
    want_l = np.maximum(((lo + hi) * rcp) >> 8, 0).astype(np.int32)
    np.testing.assert_array_equal(got_l, want_l)
