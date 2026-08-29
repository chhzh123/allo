# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A TPU with a programmable vector unit -- MXU, VPU, and an instruction stream.

`test_spmw_tpu.py` is a fixed-function engine: its activation row does ReLU and a
shift because that is what its body says, and changing the epilogue means editing
the unit and synthesising again. A real TPU does not work that way. It has a
matrix unit, a *vector* unit beside it, and a stream of instructions telling the
vector unit what to do -- so the same hardware runs bias-add-ReLU-requantise for
one layer and something else for the next.

This is that design.

    A ---> [ MXU: KT x NT weight-stationary macs ] ---> accumulators
                                                            |
    Prog -> [ VPU lane 0 ] -> [ lane 1 ] -> ... -> [ lane NT-1 ]  -> Y
              instruction chain, one lane per output column

Three things make it a processor rather than a pipeline:

* **An instruction set.** One ``int32`` per instruction, ``[opcode:8 | dst:4 |
  src:4 | imm:16]``, decoded by shifts and masks in the lane.
* **A register file.** Each lane holds ``REGS`` accumulators and instructions
  name them, so a program is a real dataflow between registers rather than a
  fixed expression.
* **An instruction buffer.** The program arrives *as data* on the chain, is
  latched once per lane, and is then replayed for every output element -- which
  is what lets a different program run without touching the hardware.

The chain is the same shape as the operand chains in
`test_spmw_autosa_match.py`: one stream in at the head, each lane keeping what it
needs and passing the rest along, so the instruction bus costs one port however
wide the array.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int8, int32

# The engine's shape: a KT x NT matrix unit fed MT rows of activations.
KT, NT, MT = 4, 4, 6
REGS = 4  # registers per VPU lane
NPROG = 8  # instruction-buffer depth, and so the program's length

# -- the instruction set ----------------------------------------------------
#
# One int32: [ opcode:8 | dst:4 | src:4 | imm:16 ]. Small enough to decode with
# two shifts and a mask, wide enough to say everything the epilogue needs.
NOP = 0
LOADZ = 1  # r[dst] = the accumulator arriving from the MXU column
LOADB = 2  # r[dst] = this lane's bias
LOADI = 3  # r[dst] = imm (sign-extended)
ADD = 4  # r[dst] += r[src]
MUL = 5  # r[dst] *= r[src]
MAX = 6  # r[dst] = max(r[dst], r[src])
SHR = 7  # r[dst] >>= imm
STORE = 8  # emit r[dst]


def encode(opcode, dst=0, src=0, imm=0):
    """One instruction word."""
    return (opcode << 24) | (dst << 20) | (src << 16) | (imm & 0xFFFF)


def program(*instructions):
    """A program padded to the instruction buffer's depth."""
    words = [encode(*i) for i in instructions]
    if len(words) > NPROG:
        raise ValueError(f"{len(words)} instructions, buffer holds {NPROG}")
    return np.array(words + [encode(NOP)] * (NPROG - len(words)), dtype=np.int32)


# bias-add, ReLU, requantise -- the epilogue the fixed-function engine hard-codes
RELU_REQUANT = program(
    (LOADZ, 0),
    (LOADB, 1),
    (ADD, 0, 1),
    (LOADI, 2, 0, 0),
    (MAX, 0, 2),
    (SHR, 0, 0, 4),
    (STORE, 0),
)

# the same hardware, a different epilogue: bias-add then double, no clamp
BIAS_DOUBLE = program(
    (LOADZ, 0),
    (LOADB, 1),
    (ADD, 0, 1),
    (LOADI, 2, 0, 2),
    (MUL, 0, 2),
    (STORE, 0),
)


class MacIO(spmw.Interface):
    """A weight-stationary multiply-accumulate cell."""

    a_in = spmw.In(int8)
    a_out = spmw.Out(int8)
    p_in = spmw.In(int32)
    p_out = spmw.Out(int32)
    w = spmw.MemIn(int8)  # this cell's stationary weight


class VpuIO(spmw.Interface):
    """One vector lane, and the instruction chain passing through it."""

    op_in = spmw.In(int32)
    op_out = spmw.Out(int32)
    z_in = spmw.In(int32)  # the accumulator this lane post-processes
    y_out = spmw.Out(int32)
    b = spmw.MemIn(int32)  # this lane's stationary bias


mxu = spmw.Topology(
    MacIO,
    grid=(KT, NT),
    link=lambda i, j: {
        MacIO.a_out: spmw.to((i, j + 1), MacIO.a_in),
        MacIO.p_out: spmw.to((i + 1, j), MacIO.p_in),
    },
)

vpu_chain = spmw.Topology(
    VpuIO,
    grid=(NT,),
    link=lambda i: {VpuIO.op_out: spmw.to((i + 1,), VpuIO.op_in)},
)


@spmw.unit
def mac(io: MacIO):
    """Unchanged from the fixed-function engine: activations east, sums south."""
    for m in range(MT):
        a = io.a_in.get()
        p = io.p_in.get()
        io.p_out.put(p + a * io.w)
        io.a_out.put(a)


@spmw.unit
def vpu(io: VpuIO):
    """One vector lane: latch the program, then run it per output element.

    The two loops are the whole architecture. The first is the instruction
    buffer filling from the chain -- paid once, whatever the tile height. The
    second is the lane executing that buffer against each accumulator arriving
    from the MXU, which is what makes the epilogue a program rather than a
    pipeline stage.
    """
    prog: int32[NPROG]
    for pc in range(NPROG):
        word: int32 = io.op_in.get()
        prog[pc] = word
        io.op_out.put(word)

    for m in range(MT):
        z: int32 = io.z_in.get()
        reg: int32[REGS]
        for step in range(NPROG):
            word2: int32 = prog[step]
            opcode: int32 = (word2 >> 24) & 255
            dst: int32 = (word2 >> 20) & 15
            src: int32 = (word2 >> 16) & 15
            imm: int32 = word2 & 65535
            if opcode == LOADZ:
                reg[dst] = z
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
def tpu(
    A: int8[MT, KT],
    W: int8[KT, NT],
    Bias: int32[NT],
    Prog: int32[NPROG],
    Y: int32[MT, NT],
):
    P = spmw.place(mac, on=mxu)
    V = spmw.place(vpu, on=vpu_chain)
    spmw.shard(W, into=P.w)
    spmw.shard(Bias, into=V.b)
    spmw.stream_in(A, into=P.a_in, index=(..., P.rows))
    spmw.stream_in(0, into=P.p_in)
    spmw.link(P.p_out, to=V.z_in)
    # The program enters the head of the chain and every lane keeps a copy.
    spmw.stream_in(Prog, into=V.op_in, index=(...,))
    (lane,) = V.axes
    spmw.gather(Y, from_=V.y_out, index=(..., lane))


# What the build script must put in `Prog` when it drives this design: an
# instruction tensor filled with random small integers is not a program.
tpu.spmw_operands = {"Prog": RELU_REQUANT}


def _reference(A, W, Bias, prog):
    """What the program means, in numpy -- an interpreter for the same ISA."""
    z_all = A.astype(np.int32) @ W.astype(np.int32)
    out = np.zeros_like(z_all)
    for m in range(z_all.shape[0]):
        for lane in range(z_all.shape[1]):
            reg = [np.int32(0)] * REGS
            z = np.int32(z_all[m, lane])
            for word in prog:
                opcode = (int(word) >> 24) & 0xFF
                dst = (int(word) >> 20) & 0xF
                src = (int(word) >> 16) & 0xF
                imm = int(word) & 0xFFFF
                if opcode == LOADZ:
                    reg[dst] = z
                elif opcode == LOADB:
                    reg[dst] = np.int32(Bias[lane])
                elif opcode == LOADI:
                    reg[dst] = np.int32(imm)
                elif opcode == ADD:
                    reg[dst] = np.int32(reg[dst] + reg[src])
                elif opcode == MUL:
                    reg[dst] = np.int32(reg[dst] * reg[src])
                elif opcode == MAX:
                    reg[dst] = np.int32(max(reg[dst], reg[src]))
                elif opcode == SHR:
                    reg[dst] = np.int32(reg[dst] >> imm)
                elif opcode == STORE:
                    out[m, lane] = reg[dst]
    return out


def _operands(seed=2, prog=None):
    """Operands wide enough that the epilogue is actually exercised.

    A requantising shift crushes small numbers: with A and W drawn from
    [-8, 8) the products land in [-256, 256] and `>> 4` after a ReLU left 3 of
    24 outputs nonzero, so a passing comparison said almost nothing. Drawing
    from [-16, 16) puts the accumulator near +/-900 and most outputs survive.
    `test_the_operands_exercise_the_program` is what keeps that true. The
    default seed is chosen for coverage rather than habit.
    """
    rng = np.random.default_rng(seed)
    A = rng.integers(-16, 16, (MT, KT)).astype(np.int8)
    W = rng.integers(-16, 16, (KT, NT)).astype(np.int8)
    Bias = rng.integers(-64, 64, (NT,)).astype(np.int32)
    return A, W, Bias, RELU_REQUANT if prog is None else prog


def test_the_operands_exercise_the_program():
    """A comparison against a mostly-zero expectation is not a test.

    Both halves of the epilogue have to be visible in the answer. About half the
    accumulators are negative and a ReLU is *supposed* to zero those, so zeros
    are not the defect -- the defect would be a shift so large that the
    survivors are crushed too, which is what drawing A and W from [-8, 8) did:
    3 of 24 outputs nonzero, 4 distinct values, everything under 5.

    So this asks for both: enough nonzeros that the arithmetic is checked, and
    at least one zero so the clamp is.
    """
    A, W, Bias, prog = _operands()
    out = _reference(A, W, Bias, prog)
    assert np.count_nonzero(out) >= out.size // 3, out
    assert np.count_nonzero(out == 0) >= 1, out
    assert len(np.unique(out)) >= 8, np.unique(out)


def test_the_isa_round_trips():
    """Encode and decode agree, which the hardware's decode depends on."""
    word = encode(SHR, dst=3, src=2, imm=4)
    assert (word >> 24) & 0xFF == SHR
    assert (word >> 20) & 0xF == 3
    assert (word >> 16) & 0xF == 2
    assert word & 0xFFFF == 4


def test_reference_matches_numpy_for_the_relu_program():
    """The ISA interpreter agrees with the closed form it is meant to compute."""
    A, W, Bias, prog = _operands()
    want = np.maximum(A.astype(np.int32) @ W.astype(np.int32) + Bias, 0) >> 4
    np.testing.assert_array_equal(_reference(A, W, Bias, prog), want)


def test_it_elaborates_into_two_placements_and_an_instruction_chain():
    graph = spmw.elaborate(tpu)
    assert len(graph.placements) == 2
    assert [b.kind for b in graph.bindings] == [
        "shard",
        "shard",
        "stream_in",
        "seed",
        "link",
        "stream_in",
        "gather",
    ]


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_it_runs_the_relu_program(target):
    A, W, Bias, prog = _operands()
    Y = np.zeros((MT, NT), dtype=np.int32)
    spmw.build(tpu, target=target)(A, W, Bias, prog, Y)
    np.testing.assert_array_equal(Y, _reference(A, W, Bias, prog))


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_the_same_hardware_runs_a_different_program(target):
    """The point of having instructions: only the data changes."""
    A, W, Bias, _ = _operands(seed=3)
    Y = np.zeros((MT, NT), dtype=np.int32)
    spmw.build(tpu, target=target)(A, W, Bias, BIAS_DOUBLE, Y)
    np.testing.assert_array_equal(Y, _reference(A, W, Bias, BIAS_DOUBLE))
    # and it really is a different answer
    assert not np.array_equal(
        _reference(A, W, Bias, BIAS_DOUBLE), _reference(A, W, Bias, RELU_REQUANT)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
