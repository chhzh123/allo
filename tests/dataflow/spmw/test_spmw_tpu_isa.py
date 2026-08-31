# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A TPU whose *matrix* unit takes instructions too.

`test_spmw_tpu_vpu.py` made the epilogue programmable and left the matmul fixed:
the MXU multiplied, every step, by the one weight each cell held. This makes the
matrix unit a consumer of instructions as well.

Each cell holds ``NW`` weights -- a small weight file, not one number -- and an
instruction says *which* to use and *what to do* with it:

    MACC  tile     p_out = p_in + a * w[tile]
    MZERO tile     p_out =        a * w[tile]     (start a fresh accumulation)
    MSKIP          p_out = p_in                   (consume the activation, add
                                                   nothing: padding and masking)

The instruction rides the same road the activation does -- west to east, one
loader on column 0 and the link carrying it across -- so it costs one port per
row however wide the array is.

What that buys is the point. Three projections of the same input against three
different weight matrices used to mean three weight loads; now it is one load
and three instructions, which is what makes a sequence of matmuls something the
engine can be *told* to do rather than rebuilt for. `test_spmw_transformer.py`
uses exactly that to run an attention block.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int8, int32

D = 4  # the array is D x D: reduction depth by output width
SEQ = 4  # rows of activations per pass
NW = 4  # weight matrices held per cell -- the cell's weight file
REGS = 4
NPROG = 12  # VPU instruction-buffer depth

# -- the MXU's instruction set ----------------------------------------------
MACC = 1
MZERO = 2
MSKIP = 3

# -- the VPU's, unchanged from test_spmw_tpu_vpu.py plus ACCZ ---------------
NOP = 0
LOADZ = 1
LOADB = 2
LOADI = 3
ADD = 4
MUL = 5
MAX = 6
SHR = 7
STORE = 8
ACCZ = 9


def mxu_word(opcode, tile=0):
    """One MXU instruction: [opcode:8 | tile:8 | 0:16]."""
    return (opcode << 24) | (tile << 16)


def vpu_word(opcode, dst=0, src=0, imm=0):
    """One VPU instruction: [opcode:8 | dst:4 | src:4 | imm:16]."""
    return (opcode << 24) | (dst << 20) | (src << 16) | (imm & 0xFFFF)


def mxu_program(steps, rows=D):
    """An MXU program, broadcast to every row of the array.

    Every row runs the same instruction at the same step, so the tensor is one
    program repeated across its columns; the loader on column 0 and the
    west-to-east link do the rest.
    """
    words = np.array([mxu_word(*s) for s in steps], dtype=np.int32)
    return np.repeat(words[:, None], rows, axis=1)


def vpu_program(instructions, depth=NPROG):
    words = [vpu_word(*i) for i in instructions]
    if len(words) > depth:
        raise ValueError(f"{len(words)} instructions, buffer holds {depth}")
    return np.array(words + [vpu_word(NOP)] * (depth - len(words)), dtype=np.int32)


class MacIO(spmw.Interface):
    """A cell with a weight file and an instruction port."""

    op_in = spmw.In(int32)
    op_out = spmw.Out(int32)
    a_in = spmw.In(int8)
    a_out = spmw.Out(int8)
    p_in = spmw.In(int32)
    p_out = spmw.Out(int32)
    w = spmw.MemIn(int8[NW])


class VpuIO(spmw.Interface):
    op_in = spmw.In(int32)
    op_out = spmw.Out(int32)
    z_in = spmw.In(int32)
    y_out = spmw.Out(int32)
    b = spmw.MemIn(int32)


def _engine(steps, vprog_len=NPROG):
    """The fabric, at a chosen number of MXU steps.

    The step count is structural -- it is how many instructions the array
    executes -- so it is a parameter of the design rather than of a call.
    """

    mxu = spmw.Topology(
        MacIO,
        grid=(D, D),
        link=lambda i, j: {
            MacIO.a_out: spmw.to((i, j + 1), MacIO.a_in),
            MacIO.op_out: spmw.to((i, j + 1), MacIO.op_in),
            MacIO.p_out: spmw.to((i + 1, j), MacIO.p_in),
        },
    )
    chain = spmw.Topology(
        VpuIO,
        grid=(D,),
        link=lambda i: {VpuIO.op_out: spmw.to((i + 1,), VpuIO.op_in)},
    )

    @spmw.unit
    def mac(io: MacIO):
        for step in range(steps):
            word: int32 = io.op_in.get()
            io.op_out.put(word)
            opcode: int32 = (word >> 24) & 255
            tile: int32 = (word >> 16) & 255
            # The activation is consumed whatever the instruction says, so a
            # skipped step is a bubble rather than a desynchronised stream.
            a = io.a_in.get()
            p = io.p_in.get()
            io.a_out.put(a)
            wt: int32 = io.w[tile]
            if opcode == MACC:
                io.p_out.put(p + a * wt)
            elif opcode == MZERO:
                io.p_out.put(a * wt)
            else:
                io.p_out.put(p)

    @spmw.unit
    def vpu(io: VpuIO):
        prog: int32[vprog_len]
        for pc in range(vprog_len):
            word: int32 = io.op_in.get()
            prog[pc] = word
            io.op_out.put(word)

        for m in range(steps):
            reg: int32[REGS]
            for pc2 in range(vprog_len):
                word2: int32 = prog[pc2]
                opcode: int32 = (word2 >> 24) & 255
                dst: int32 = (word2 >> 20) & 15
                src: int32 = (word2 >> 16) & 15
                imm: int32 = word2 & 65535
                if opcode == ACCZ:
                    zz: int32 = io.z_in.get()
                    reg[dst] = reg[dst] + zz
                elif opcode == LOADB:
                    reg[dst] = io.b
                elif opcode == LOADI:
                    reg[dst] = imm
                elif opcode == ADD:
                    reg[dst] = reg[dst] + reg[src]
                elif opcode == MUL:
                    reg[dst] = reg[dst] * reg[src]
                elif opcode == MAX:
                    if reg[src] > reg[dst]:
                        reg[dst] = reg[src]
                elif opcode == SHR:
                    reg[dst] = reg[dst] >> imm
                elif opcode == STORE:
                    io.y_out.put(reg[dst])

    @spmw.fabric
    def engine(
        A: int8[steps, D],
        W: int8[D, D, NW],
        Bias: int32[D],
        MProg: int32[steps, D],
        VProg: int32[vprog_len],
        Y: int32[steps, D],
    ):
        P = spmw.place(mac, on=mxu)
        V = spmw.place(vpu, on=chain)
        spmw.shard(W, into=P.w)  # cell (k, c) holds W[k, c, :]
        spmw.shard(Bias, into=V.b)
        spmw.stream_in(A, into=P.a_in, index=(..., P.rows))
        spmw.stream_in(MProg, into=P.op_in, index=(..., P.rows))
        spmw.stream_in(0, into=P.p_in)
        spmw.link(P.p_out, to=V.z_in)
        spmw.stream_in(VProg, into=V.op_in, index=(...,))
        (lane,) = V.axes
        spmw.gather(Y, from_=V.y_out, index=(..., lane))

    engine.spmw_parts = (mac, vpu, steps, vprog_len)
    return engine


# One pass of SEQ rows: the plain matmul, so the instruction-driven engine can
# be checked against the fixed one it replaces.
ONE_PASS = _engine(SEQ)

# A VPU program that just moves the accumulator out, so the test sees the MXU's
# arithmetic and nothing else.
PASSTHROUGH = vpu_program([(ACCZ, 0), (STORE, 0)])


# What the build script must load: an instruction tensor of random small
# integers is not a program. A plain matmul on tile 1, moved out unchanged, so a
# cosim failure points at the hardware rather than at a clever epilogue.
ONE_PASS.spmw_operands = {
    "MProg": mxu_program([(MACC, 1)] * SEQ),
    "VProg": PASSTHROUGH,
}


def mxu_reference(A, W, mprog_col):
    """What the MXU computes, stepped exactly as the hardware steps it.

    Written against the instruction stream rather than as a matmul, because that
    is the thing under test: the answer is a matmul only when the program says
    so.
    """
    steps = A.shape[0]
    out = np.zeros((steps, W.shape[1]), dtype=np.int64)
    for step in range(steps):
        word = int(mprog_col[step])
        opcode = (word >> 24) & 0xFF
        tile = (word >> 16) & 0xFF
        for c in range(W.shape[1]):
            acc = np.int64(0)
            for k in range(W.shape[0]):
                a = np.int64(A[step, k])
                wt = np.int64(W[k, c, tile])
                if opcode == MACC:
                    acc += a * wt
                elif opcode == MZERO:
                    # Only the last row's contribution survives a fresh start,
                    # because every cell below restarts the column too.
                    acc = a * wt
            out[step, c] = acc if opcode != MSKIP else 0
    return out


def _operands(seed=1, steps=SEQ):
    rng = np.random.default_rng(seed)
    A = rng.integers(-16, 16, (steps, D)).astype(np.int8)
    W = rng.integers(-16, 16, (D, D, NW)).astype(np.int8)
    Bias = np.zeros(D, dtype=np.int32)
    return A, W, Bias


def _run(engine, A, W, Bias, mprog, vprog, steps=SEQ, target="ref"):
    Y = np.zeros((steps, D), dtype=np.int32)
    spmw.build(engine, target=target)(A, W, Bias, mprog, vprog, Y)
    return Y


def test_the_mxu_isa_round_trips():
    word = mxu_word(MACC, tile=3)
    assert (word >> 24) & 0xFF == MACC
    assert (word >> 16) & 0xFF == 3


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_macc_is_the_matmul_it_replaces(target):
    """With every step a MACC on tile 0, the engine is the plain matmul."""
    A, W, Bias = _operands()
    mprog = mxu_program([(MACC, 0)] * SEQ)
    Y = _run(ONE_PASS, A, W, Bias, mprog, PASSTHROUGH, target=target)
    want = A.astype(np.int32) @ W[:, :, 0].astype(np.int32)
    np.testing.assert_array_equal(Y, want)


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_the_instruction_selects_the_weight_matrix(target):
    """One weight load, three different projections -- the point of the tile field.

    Q, K and V come out of the same silicon holding the same weights, differing
    only in which instruction each step issues.
    """
    A, W, Bias = _operands()
    for tile in range(3):
        mprog = mxu_program([(MACC, tile)] * SEQ)
        Y = _run(ONE_PASS, A, W, Bias, mprog, PASSTHROUGH, target=target)
        want = A.astype(np.int32) @ W[:, :, tile].astype(np.int32)
        np.testing.assert_array_equal(Y, want)
    # and the three really are different matrices
    assert not np.array_equal(W[:, :, 0], W[:, :, 1])


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_mskip_masks_a_step(target):
    """A skipped step consumes its activation and contributes nothing.

    That is what padding a sequence to the array's height costs: a bubble, not a
    wrong answer.
    """
    A, W, Bias = _operands()
    mprog = mxu_program([(MACC, 0), (MSKIP,), (MACC, 0), (MSKIP,)])
    Y = _run(ONE_PASS, A, W, Bias, mprog, PASSTHROUGH, target=target)
    want = A.astype(np.int32) @ W[:, :, 0].astype(np.int32)
    np.testing.assert_array_equal(Y[0], want[0])
    np.testing.assert_array_equal(Y[1], np.zeros(D, dtype=np.int32))
    np.testing.assert_array_equal(Y[2], want[2])
    np.testing.assert_array_equal(Y[3], np.zeros(D, dtype=np.int32))


def test_the_reference_agrees_with_numpy_for_a_plain_matmul():
    """The step-by-step model and the closed form must not disagree."""
    A, W, _ = _operands()
    mprog = mxu_program([(MACC, 1)] * SEQ)
    np.testing.assert_array_equal(
        mxu_reference(A, W, mprog[:, 0]),
        A.astype(np.int64) @ W[:, :, 1].astype(np.int64),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
