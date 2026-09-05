# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine that runs a whole GEMM stage in one launch.

`test_spmw_tpu_isa.py`'s engine holds NW=4 weight tiles per cell and runs one
instruction per step, so a K=1024 reduction is 64 separate launches per 16x16
output tile and the array idles behind the PCIe round trip 99.7% of the time.
This is the same MXU and VPU with two changes that are instructions rather
than structure:

* the cell's weight file is ``kfile`` tiles deep -- 256 holds K=4096, or four
  16-column output tiles of K=1024 -- and
* ``MSWEEP base count`` makes the cell run ``count`` steps selecting tiles
  ``base .. base+count-1`` in turn, so one word is a whole reduction, and the
  lane's ``ACCN n`` folds ``n`` partial sums at II=1 instead of one per
  dispatched instruction.

A launch therefore produces ``outs`` result rows each the full-K dot product
of an activation row with a 16-column weight slab, and the host walks a GPT-2
layer as a few hundred launches of ~100 us of array work rather than four
hundred thousand of 64 ns.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int8, int32

from test_spmw_tpu_isa import (
    ACCZ,
    ADD,
    EXP2,
    LOADB,
    LOADI,
    LOADR,
    LOADZ,
    MAX,
    MUL,
    NB,
    NOP,
    NPROG,
    RCP_BITS,
    REGS,
    SHR,
    STORE,
    SUB,
    VpuIO,
    vpu_header,
    vpu_word,
)

# -- the MXU's instruction set, sweep form -----------------------------------
MSWEEP = 4  # [opcode:8 | base:8 | count:16]: MACC over tiles base..base+count-1
MPASS = 5  # [opcode:8 | 0:8 | count:16]: count steps that forward the psum
MLOAD = 6  # [opcode:8 | 0:8 | words:16]: fill the file from the weight stream

# -- the VPU's extra opcode --------------------------------------------------
ACCN = 13  # [opcode:8 | dst:4 | 0:4 | n:16]: reg[dst] += the next n psums


def mxu_sweep(base, count):
    return (MSWEEP << 24) | (base << 16) | (count & 0xFFFF)


def mxu_pass(count):
    return (MPASS << 24) | (count & 0xFFFF)


def mxu_load(words):
    return (MLOAD << 24) | (words & 0xFFFF)


def file_to_stream(W):
    """The weight file `W[k, c, :]` as the stream row `k` reads, packed.

    Weights reach a cell as a stream of 32-bit words along its row, four int8
    per word, not as one wide token: the first cell of a row keeps the first
    ``kfile // 4`` words and forwards the rest, so cell `c` ends up holding
    ``W[k, c, :]``. Delivered this way the file is a block RAM in the cell
    and the mesh carries 32-bit links, where the token form was 2,048
    flip-flops a cell and a mesh Vivado could not route.
    """
    dim, _, kfile = W.shape
    rows = W.reshape(dim, dim * kfile).astype(np.uint8).astype(np.uint32)
    words = (
        rows[:, 0::4]
        | (rows[:, 1::4] << 8)
        | (rows[:, 2::4] << 16)
        | (rows[:, 3::4] << 24)
    )
    return words.astype(np.int32).T.copy()  # [dim * kfile // 4, dim]


def mxu_program(words, rows):
    """[count] + words, broadcast across the array's rows."""
    body = [len(words)] + list(words)
    return np.repeat(np.array(body, dtype=np.int32)[:, None], rows, axis=1)


def stage_vprog(sweep, shift, outs, bias=False, depth=NPROG):
    """Fold `sweep` partial sums per output row, rescale, emit `outs` rows."""
    prog = [(LOADB, 0, 0, 0) if bias else (LOADI, 0, 0, 0)]
    prog += [(ACCN, 0, 0, sweep), (SHR, 0, 0, shift), (STORE, 0, 0, 0)]
    words = [vpu_word(*p) for p in prog]
    body = words + [vpu_word(NOP)] * (depth - len(words))
    return np.array([vpu_header(outs, len(words))] + body, dtype=np.int32)


class MacIO(spmw.Interface):
    """A cell with a deep weight file and an instruction port."""

    op_in = spmw.In(int32)
    op_out = spmw.Out(int32)
    a_in = spmw.In(int8)
    a_out = spmw.Out(int8)
    p_in = spmw.In(int32)
    p_out = spmw.Out(int32)
    w_in = spmw.In(int32)
    w_out = spmw.Out(int32)


def stage_engine(dim, kfile, outs, sweep, words=None, vprog_len=NPROG):
    """The fabric for one launch: `outs` result rows, each a `sweep`-tile sum.

    ``words`` is the MXU program length the buffer is built for; it defaults to
    one sweep word per output row. ``steps`` -- the activation stream -- is
    ``outs * sweep``.
    """
    words = outs + 1 if words is None else words  # the load word, then a sweep per row
    steps = outs * sweep
    kw = kfile // 4  # the file, in 32-bit words

    class CellIO(spmw.Interface):
        op_in = spmw.In(int32)
        op_out = spmw.Out(int32)
        a_in = spmw.In(int8)
        a_out = spmw.Out(int8)
        p_in = spmw.In(int32)
        p_out = spmw.Out(int32)
        w_in = spmw.In(int32)
        w_out = spmw.Out(int32)

    mxu = spmw.Topology(
        CellIO,
        grid=(dim, dim),
        link=lambda i, j: {
            CellIO.a_out: spmw.to((i, j + 1), CellIO.a_in),
            CellIO.op_out: spmw.to((i, j + 1), CellIO.op_in),
            CellIO.w_out: spmw.to((i, j + 1), CellIO.w_in),
            CellIO.p_out: spmw.to((i + 1, j), CellIO.p_in),
        },
    )
    chain = spmw.Topology(
        VpuIO,
        grid=(dim,),
        link=lambda i: {VpuIO.op_out: spmw.to((i + 1,), VpuIO.op_in)},
    )

    @spmw.unit
    def mac(io: CellIO):
        # The weight file, filled by a load word from the row's weight stream
        # and read by the sweeps that follow it. A local array, so HLS gives
        # it a block RAM rather than the flip-flops a port token would be.
        wf: int32[kw]
        count: int32 = io.op_in.get()
        io.op_out.put(count)
        for _w in range(count):
            word: int32 = io.op_in.get()
            opcode: int32 = (word >> 24) & 255
            base: int32 = (word >> 16) & 255
            n: int32 = word & 65535
            if opcode == MLOAD:
                # `n` words are arriving for this cell and the ones beyond it:
                # keep the file's worth, pass the rest on, and tell the next
                # cell how many that was. Every cell runs the same program.
                io.op_out.put((MLOAD << 24) | (n - kw))
                for i in range(kw):
                    wf[i] = io.w_in.get()
                for _j in range(n - kw):
                    x: int32 = io.w_in.get()
                    io.w_out.put(x)
            else:
                io.op_out.put(word)
                # One word, many steps: the activation and the partial sum are
                # consumed every step whatever the opcode, so the streams stay
                # in lockstep with the lane that counts them.
                for t in range(n):
                    a = io.a_in.get()
                    p = io.p_in.get()
                    io.a_out.put(a)
                    if opcode == MSWEEP:
                        idx: int32 = base + t
                        packed: int32 = wf[idx >> 2]
                        byte: int32 = (packed >> ((idx & 3) * 8)) & 255
                        wt: int32 = (byte ^ 128) - 128
                        io.p_out.put(p + a * wt)
                    else:
                        io.p_out.put(p)

    @spmw.unit
    def vpu(io: VpuIO):
        header: int32 = io.op_in.get()
        io.op_out.put(header)
        plen: int32 = header & 65535
        # How many rows this launch emits is in the header, so FFN2 -- 128
        # rows of a 256-tile sweep -- and a projection -- 512 rows of a
        # 64-tile sweep -- are the same 32,768 activation steps on the same
        # netlist, told apart by two numbers in the stream.
        nouts: int32 = (header >> 16) & 65535
        prog: int32[vprog_len]
        for pc in range(plen):
            word: int32 = io.op_in.get()
            prog[pc] = word
            io.op_out.put(word)
        for _pad in range(vprog_len - plen):
            spare: int32 = io.op_in.get()
            io.op_out.put(spare)

        # The pass reciprocal, once per launch -- see test_spmw_tpu_isa for why
        # it does not belong inside the dispatch.
        denom: int32 = io.b[1]
        rcp: int32 = 0
        if denom > 0:
            rcp = (1 << RCP_BITS) // denom

        # The register file survives from one row to the next, so a program can
        # reduce down a lane's rows -- a running maximum, a running sum -- which
        # is how softmax runs on the same netlist as the GEMMs.
        reg: int32[REGS]
        for r0 in range(REGS):
            reg[r0] = 0
        for _m in range(nouts):
            for pc2 in range(plen):
                word2: int32 = prog[pc2]
                opcode: int32 = (word2 >> 24) & 255
                dst: int32 = (word2 >> 20) & 15
                src: int32 = (word2 >> 16) & 15
                imm: int32 = word2 & 65535
                if opcode == ACCN:
                    # The whole reduction, folded at one psum per cycle. This
                    # is the instruction that lets the lane keep pace with the
                    # array; dispatching one ACCZ per psum ran at II=2 and made
                    # the vector unit the bottleneck of a matrix engine.
                    acc: int32 = reg[dst]
                    for _i in range(imm):
                        zz: int32 = io.z_in.get()
                        acc = acc + zz
                    reg[dst] = acc
                elif opcode == ACCZ:
                    z1: int32 = io.z_in.get()
                    reg[dst] = reg[dst] + z1
                elif opcode == LOADZ:
                    z2: int32 = io.z_in.get()
                    reg[dst] = z2
                elif opcode == LOADB:
                    reg[dst] = io.b[src]
                elif opcode == LOADI:
                    reg[dst] = imm
                elif opcode == ADD:
                    reg[dst] = reg[dst] + reg[src]
                elif opcode == SUB:
                    reg[dst] = reg[dst] - reg[src]
                elif opcode == MUL:
                    reg[dst] = reg[dst] * reg[src]
                elif opcode == MAX:
                    if reg[src] > reg[dst]:
                        reg[dst] = reg[src]
                elif opcode == SHR:
                    reg[dst] = reg[dst] >> imm
                elif opcode == EXP2:
                    e: int32 = reg[dst]
                    if e < 0:
                        e = 0
                    if e > 30:
                        e = 30
                    reg[dst] = 1 << e
                elif opcode == LOADR:
                    reg[dst] = rcp
                elif opcode == STORE:
                    io.y_out.put(reg[dst])

    @spmw.fabric
    def engine(
        A: int8[steps, dim],
        W: int32[dim * kw, dim],
        Bias: int32[dim, NB],
        MProg: int32[words + 1, dim],
        VProg: int32[vprog_len + 1],
        Y: int32[outs, dim],
    ):
        P = spmw.place(mac, on=mxu)
        V = spmw.place(vpu, on=chain)
        spmw.shard(Bias, into=V.b)
        spmw.stream_in(A, into=P.a_in, index=(..., P.rows))
        spmw.stream_in(
            W, into=P.w_in, index=(..., P.rows)
        )  # cell (k, c) keeps W[k, c, :]
        spmw.stream_in(MProg, into=P.op_in, index=(..., P.rows))
        spmw.stream_in(0, into=P.p_in)
        spmw.link(P.p_out, to=V.z_in)
        spmw.stream_in(VProg, into=V.op_in, index=(...,))
        (lane,) = V.axes
        spmw.gather(Y, from_=V.y_out, index=(..., lane))

    engine.spmw_parts = (mac, vpu, dim, kfile, outs, sweep, words, steps)
    return engine


# -- laying a GEMM stage out for the engine ----------------------------------


def stage_operands(X, Wmat, dim, kfile, shift, slabs=None, rng=None):
    """One launch of `X[R, K] @ Wmat[K, N]` for `slabs` 16-column slabs of N.

    Returns (A, W, Bias, MProg, VProg, expected). `A` is the activation stream
    the array reads: for each slab, each row, each K-tile, one 16-wide vector.
    `W` is the cell weight file as the rows stream it in (`file_to_stream`):
    tile `s*T + t` is the (t, s) 16x16 block. The program is one load word,
    then a sweep per output row.
    """
    R, K = X.shape
    N = Wmat.shape[1]
    T = K // dim
    slabs = N // dim if slabs is None else slabs
    assert slabs * T <= kfile, f"{slabs} slabs of {T} tiles exceed the {kfile} file"
    outs = slabs * R
    A = np.zeros((outs * T, dim), dtype=np.int8)
    W = np.zeros((dim, dim, kfile), dtype=np.int8)
    words = []
    for s in range(slabs):
        for t in range(T):
            W[:, :, s * T + t] = Wmat[t * dim : (t + 1) * dim, s * dim : (s + 1) * dim]
        for r in range(R):
            row = s * R + r
            for t in range(T):
                A[row * T + t] = X[r, t * dim : (t + 1) * dim]
            words.append(mxu_sweep(s * T, T))
    mprog = mxu_program([mxu_load(dim * kfile // 4)] + words, dim)
    vprog = stage_vprog(T, shift, outs)
    bias = np.zeros((dim, NB), dtype=np.int32)
    full = X.astype(np.int32) @ Wmat.astype(np.int32)
    expected = np.zeros((outs, dim), dtype=np.int32)
    for s in range(slabs):
        expected[s * R : (s + 1) * R] = full[:, s * dim : (s + 1) * dim] >> shift
    return A, file_to_stream(W), bias, mprog, vprog, expected


def _case(dim, K, N, R, kfile, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(-8, 8, size=(R, K)).astype(np.int8)
    Wm = rng.integers(-4, 4, size=(K, N)).astype(np.int8)
    return X, Wm


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_one_slab_full_k(target):
    """A K=32 reduction on a 4x4 array: 8 tiles swept in one word per row."""
    dim, K, N, R, kfile = 4, 32, 4, 6, 16
    X, Wm = _case(dim, K, N, R, kfile)
    A, W, bias, mprog, vprog, want = stage_operands(X, Wm, dim, kfile, shift=0)
    eng = stage_engine(dim, kfile, outs=R, sweep=K // dim)
    Y = np.zeros((R, dim), dtype=np.int32)
    spmw.build(eng, target=target)(A, W, bias, mprog, vprog, Y)
    np.testing.assert_array_equal(Y, want)


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_two_slabs_one_launch(target):
    """Two 16-column output slabs from one weight file, one launch."""
    dim, K, N, R, kfile = 4, 16, 8, 5, 16
    X, Wm = _case(dim, K, N, R, kfile, seed=3)
    A, W, bias, mprog, vprog, want = stage_operands(X, Wm, dim, kfile, shift=1)
    eng = stage_engine(dim, kfile, outs=2 * R, sweep=K // dim)
    Y = np.zeros((2 * R, dim), dtype=np.int32)
    spmw.build(eng, target=target)(A, W, bias, mprog, vprog, Y)
    np.testing.assert_array_equal(Y, want)


def test_a_launch_may_be_smaller_than_the_buffer():
    """FFN2 and a projection are the same 32,768 steps; only the counts move.

    The reference simulator feeds whole tensors, so "fewer rows than the
    buffer" cannot be run through it -- an over-supplied stream is, correctly,
    a deadlock there. What can be checked is that every board shape fits the
    one netlist and says so in its counts.
    """
    dim, kfile, rows = 16, 256, 128
    proj = stage_operands(
        np.zeros((rows, 1024), np.int8), np.zeros((1024, 64), np.int8), dim, kfile, 4
    )
    ffn2 = stage_operands(
        np.zeros((rows, 4096), np.int8), np.zeros((4096, 16), np.int8), dim, kfile, 4
    )
    assert proj[0].shape[0] == ffn2[0].shape[0] == 32768  # activation steps
    assert proj[5].shape[0] == 512 and ffn2[5].shape[0] == 128  # rows emitted
    assert proj[3].shape[0] - 2 == 512 and ffn2[3].shape[0] - 2 == 128  # sweeps
    header_rows = lambda v: int(v[0]) >> 16  # noqa: E731
    assert header_rows(proj[4]) == 512 and header_rows(ffn2[4]) == 128


def running_max_vprog(sweep, outs, depth=NPROG):
    """Per row: fold the sweep into r1, keep the running max in r0, emit r0."""
    prog = [(LOADI, 1, 0, 0), (ACCN, 1, 0, sweep), (MAX, 0, 1, 0), (STORE, 0, 0, 0)]
    words = [vpu_word(*p) for p in prog]
    body = words + [vpu_word(NOP)] * (depth - len(words))
    return np.array([vpu_header(outs, len(words))] + body, dtype=np.int32)


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_a_reduction_down_the_rows(target):
    """A softmax needs a running maximum over a lane's rows; the register file
    carries it, so the same netlist that sweeps a GEMM does the reduction."""
    dim, K, N, R, kfile = 4, 16, 4, 6, 16
    X, Wm = _case(dim, K, N, R, kfile, seed=9)
    A, W, bias, mprog, _v, full = stage_operands(X, Wm, dim, kfile, shift=0)
    vprog = running_max_vprog(K // dim, R)
    eng = stage_engine(dim, kfile, outs=R, sweep=K // dim)
    Y = np.zeros((R, dim), dtype=np.int32)
    spmw.build(eng, target=target)(A, W, bias, mprog, vprog, Y)
    # r0 starts wherever the register file did, so compare from the second
    # row: a prefix maximum of the true dot products, per lane.
    want = np.maximum.accumulate(full, axis=0)
    np.testing.assert_array_equal(Y[1:], np.maximum(want[1:], Y[0]))


def test_roles_do_not_grow_with_the_file():
    """A deeper weight file is data, not structure: the same roles either way."""
    small = spmw.elaborate(stage_engine(4, 8, outs=4, sweep=2))
    deep = spmw.elaborate(stage_engine(4, 256, outs=4, sweep=2))
    assert len(small.bindings) == len(deep.bindings)


# -- attention on the same netlist -------------------------------------------
#
# The transformer block's integer softmax, unchanged: scores in QUANT_SCORE
# fixed point, exp as a shift, a per-row reciprocal. Scores are computed
# *transposed* -- K as the activations, Q^T as the weights -- so a lane holds
# one query and its keys come down the rows, and the three softmax passes are
# reductions down a lane through the identity tile.

QUANT_SCORE = 6
EXP_SHIFT = 5
EXP_BASE = 8
PROB_BITS = 6

_EXP = [
    (LOADZ, 1, 0, 0),  # r1 = this row's score
    (LOADB, 2, 0, 0),  # r2 = the lane's maximum
    (SUB, 1, 2, 0),
    (SHR, 1, 0, EXP_SHIFT),
    (LOADI, 2, 0, EXP_BASE),
    (ADD, 1, 2, 0),
    (EXP2, 1, 0, 0),
]


def _vprog(prog, outs, depth=NPROG):
    words = [vpu_word(*p) for p in prog]
    body = words + [vpu_word(NOP)] * (depth - len(words))
    return np.array([vpu_header(outs, len(words))] + body, dtype=np.int32)


def row_max_vprog(outs):
    """r0 = max over the rows so far; the last row emitted is the maximum."""
    return _vprog([(LOADZ, 1, 0, 0), (MAX, 0, 1, 0), (STORE, 0, 0, 0)], outs)


def row_sum_vprog(outs):
    """r0 += exp(score - max); the last row emitted is the sum."""
    return _vprog(_EXP + [(ADD, 0, 1, 0), (STORE, 0, 0, 0)], outs)


def normalise_vprog(outs):
    """exp(score - max) * (1 / sum), in PROB_BITS fixed point, every row."""
    return _vprog(
        _EXP
        + [
            (LOADR, 3, 0, 0),
            (MUL, 1, 3, 0),
            (SHR, 1, 0, RCP_BITS - PROB_BITS),
            (STORE, 1, 0, 0),
        ],
        outs,
    )


def identity_file(dim, kfile):
    W = np.zeros((dim, dim, kfile), dtype=np.int8)
    W[:, :, 0] = np.eye(dim, dtype=np.int8)
    return W


def pass_operands(rows_in, dim, kfile, vprog, bias):
    """One softmax pass: `rows_in` [R, dim] int8 through the identity tile."""
    R = rows_in.shape[0]
    A = rows_in.astype(np.int8).copy()
    W = file_to_stream(identity_file(dim, kfile))
    mprog = mxu_program([mxu_load(dim * kfile // 4)] + [mxu_sweep(0, 1)] * R, dim)
    return A, W, bias.astype(np.int32), mprog, vprog


def attention_head_ref(Q, K, V):
    """The integer reference for one head, step for step with the passes."""
    i32 = np.int32
    clip8 = lambda x: np.clip(x, -128, 127).astype(np.int8)  # noqa: E731
    scores = clip8((K.astype(i32) @ Q.T.astype(i32)) >> QUANT_SCORE)  # [key, query]
    row_max = np.maximum(scores.max(axis=0), 0).astype(i32)
    arg = np.clip(EXP_BASE + ((scores.astype(i32) - row_max) >> EXP_SHIFT), 0, 30)
    exps = (np.int32(1) << arg).astype(i32)
    row_sum = exps.sum(axis=0)
    recip = np.where(
        row_sum > 0, (np.int32(1) << RCP_BITS) // np.maximum(row_sum, 1), 0
    )
    probs_t = (exps * recip) >> (RCP_BITS - PROB_BITS)
    probs = clip8(probs_t.T)  # [query, key]
    attn = clip8((probs.astype(i32) @ V.astype(i32)) >> PROB_BITS)
    return scores, row_max, row_sum, probs, attn


def attention_head(Q, K, V, dim, kfile, target="ref"):
    """One head of attention as stage-engine launches, on `target`.

    Q, K: [seq, head]; V: [seq, head]. Returns (probs, attn) as the device
    would deliver them, having checked every intermediate against the integer
    reference on the way.
    """
    seq, head = Q.shape
    ref = attention_head_ref(Q, K, V)
    # 1. scores^T = K . Q^T, one launch per group of `slabs` x dim queries.
    T = head // dim
    # As many query slabs per launch as the file, the row buffer and the
    # sequence allow; the engine and the operands must agree on this number.
    slabs = max(1, min(kfile // T, 4, seq // dim))
    per_launch = slabs * dim
    scores = np.zeros((seq, seq), dtype=np.int8)  # [key, query]
    eng = stage_engine(dim, kfile, outs=slabs * seq, sweep=T)
    run = spmw.build(eng, target=target)
    for q0 in range(0, seq, per_launch):
        A, W, bias, mprog, vprog, _ = stage_operands(
            K, Q.T[:, q0 : q0 + per_launch], dim, kfile, shift=QUANT_SCORE
        )
        Y = np.zeros((slabs * seq, dim), dtype=np.int32)
        run(A, W, bias, mprog, vprog, Y)
        for s_ in range(slabs):
            scores[:, q0 + s_ * dim : q0 + (s_ + 1) * dim] = np.clip(
                Y[s_ * seq : (s_ + 1) * seq], -128, 127
            )
    np.testing.assert_array_equal(scores, ref[0])
    # 2-4. softmax down each lane, `dim` queries per launch, three passes.
    eng1 = stage_engine(dim, kfile, outs=seq, sweep=1)
    run1 = spmw.build(eng1, target=target)
    probs_t = np.zeros((seq, seq), dtype=np.int32)
    for q0 in range(0, seq, dim):
        rows = scores[:, q0 : q0 + dim]  # [key, query-lane]
        bias = np.zeros((dim, NB), dtype=np.int32)
        A, W, b, mprog, vprog = pass_operands(
            rows, dim, kfile, row_max_vprog(seq), bias
        )
        Y = np.zeros((seq, dim), dtype=np.int32)
        run1(A, W, b, mprog, vprog, Y)
        maxes = Y[-1]
        np.testing.assert_array_equal(maxes, ref[1][q0 : q0 + dim])
        bias[:, 0] = maxes
        A, W, b, mprog, vprog = pass_operands(
            rows, dim, kfile, row_sum_vprog(seq), bias
        )
        run1(A, W, b, mprog, vprog, Y)
        sums = Y[-1]
        np.testing.assert_array_equal(sums, ref[2][q0 : q0 + dim])
        bias[:, 1] = sums
        A, W, b, mprog, vprog = pass_operands(
            rows, dim, kfile, normalise_vprog(seq), bias
        )
        run1(A, W, b, mprog, vprog, Y)
        probs_t[:, q0 : q0 + dim] = Y
    probs = np.clip(probs_t.T, -128, 127).astype(np.int8)
    np.testing.assert_array_equal(probs, ref[3])
    # 5. context = P . V, one launch (head <= slabs*dim here).
    T2 = seq // dim
    slabs2 = max(1, min(kfile // T2, 4, head // dim))
    eng2 = stage_engine(dim, kfile, outs=slabs2 * seq, sweep=T2)
    run2 = spmw.build(eng2, target=target)
    attn = np.zeros((seq, head), dtype=np.int8)
    for h0 in range(0, head, slabs2 * dim):
        A, W, bias, mprog, vprog, _ = stage_operands(
            probs, V[:, h0 : h0 + slabs2 * dim], dim, kfile, shift=PROB_BITS
        )
        Y = np.zeros((slabs2 * seq, dim), dtype=np.int32)
        run2(A, W, bias, mprog, vprog, Y)
        for s_ in range(slabs2):
            attn[:, h0 + s_ * dim : h0 + (s_ + 1) * dim] = np.clip(
                Y[s_ * seq : (s_ + 1) * seq], -128, 127
            )
    np.testing.assert_array_equal(attn, ref[4])
    return probs, attn


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_one_attention_head_on_the_stage_engine(target):
    """Scores, a three-pass softmax and the context, all on the stage engine,
    bit-exact against the transformer block's integer reference."""
    dim, seq, head, kfile = 4, 8, 8, 16
    rng = np.random.default_rng(11)
    Q = rng.integers(-64, 64, (seq, head)).astype(np.int8)
    K = rng.integers(-64, 64, (seq, head)).astype(np.int8)
    V = rng.integers(-8, 8, (seq, head)).astype(np.int8)
    attention_head(Q, K, V, dim, kfile, target=target)


def gpt_stage_of(size, kfile=256, rows=128, sweep=64, slabs=4):
    """The board shape: a `size` x `size` array, K=1024 in one sweep, four
    output slabs per launch -- 512 result rows, 32,768 activation steps."""
    return stage_engine(size, kfile, outs=slabs * rows, sweep=sweep)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
