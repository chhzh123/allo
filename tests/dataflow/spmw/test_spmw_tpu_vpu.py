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


# ---------------------------------------------------------------------------
# K-tiling: a matmul deeper than the array
# ---------------------------------------------------------------------------
#
# The engine above computes A[MT, KT] @ W[KT, NT] and nothing larger: the
# reduction depth is the array's depth. A real layer is deeper than any array,
# so a TPU walks the reduction in tiles and accumulates across passes. That is
# not a new mechanism here -- it is one more instruction.

NTILE = 2  # weight tiles, so the reduction depth is NTILE * KT
NPROG_T = 12  # a tiled program is longer: one ACCZ per tile

ACCZ = 9  # r[dst] += the next accumulator from the MXU column


def tiled_program(*instructions):
    """A program for the tiled engine, checked against the hardware's contract.

    The lane consumes an accumulator exactly when it executes ``ACCZ``, so a
    program with the wrong number of them does not compute the wrong answer --
    it deadlocks against an MXU that is still producing, or hangs waiting for
    one that has stopped. That is a programming error worth catching here
    rather than in a waveform.
    """
    words = [encode(*i) for i in instructions]
    accs = sum(1 for i in instructions if i[0] == ACCZ)
    if accs != NTILE:
        raise ValueError(f"{accs} ACCZ instructions, the MXU emits {NTILE} per output")
    if len(words) > NPROG_T:
        raise ValueError(f"{len(words)} instructions, buffer holds {NPROG_T}")
    return np.array(words + [encode(NOP)] * (NPROG_T - len(words)), dtype=np.int32)


TILED_RELU = tiled_program(
    (LOADI, 0, 0, 0),  # r0 = 0, the reduction accumulator
    (ACCZ, 0),  # += tile 0's partial sum
    (ACCZ, 0),  # += tile 1's
    (LOADB, 1),
    (ADD, 0, 1),
    (LOADI, 2, 0, 0),
    (MAX, 0, 2),
    (SHR, 0, 0, 4),
    (STORE, 0),
)


class TiledMacIO(spmw.Interface):
    """A cell holding one weight per tile rather than one weight."""

    a_in = spmw.In(int8)
    a_out = spmw.Out(int8)
    p_in = spmw.In(int32)
    p_out = spmw.Out(int32)
    # One weight per tile. A block-valued memory port takes the block axis
    # *last* in the tensor it is sharded from, so the weights arrive laid out
    # per cell -- `W[k, c, t]` -- rather than as the mathematical `W[k', c]`.
    # A real TPU's host pre-tiles its weights for the same reason.
    w = spmw.MemIn(int8[NTILE])


class TiledVpuIO(spmw.Interface):
    op_in = spmw.In(int32)
    op_out = spmw.Out(int32)
    z_in = spmw.In(int32)
    y_out = spmw.Out(int32)
    b = spmw.MemIn(int32)


tiled_mxu = spmw.Topology(
    TiledMacIO,
    grid=(KT, NT),
    link=lambda i, j: {
        TiledMacIO.a_out: spmw.to((i, j + 1), TiledMacIO.a_in),
        TiledMacIO.p_out: spmw.to((i + 1, j), TiledMacIO.p_in),
    },
)

tiled_chain = spmw.Topology(
    TiledVpuIO,
    grid=(NT,),
    link=lambda i: {TiledVpuIO.op_out: spmw.to((i + 1,), TiledVpuIO.op_in)},
)


@spmw.unit
def tiled_mac(io: TiledMacIO):
    """The same cell, walking its weight vector.

    `m` outside and `t` inside is not arbitrary: it makes a column emit all of
    one output's partial sums before moving on, which is the order the lane's
    `ACCZ` instructions consume them in. The other nesting would need a buffer
    in every lane.
    """
    for m in range(MT):
        for t in range(NTILE):
            a = io.a_in.get()
            p = io.p_in.get()
            # Widened before the multiply, not after. Two int8s can product to
            # 225, which does not fit in an int8: C++ promotes and gets this
            # right by accident, the reference simulator keeps numpy's int8 and
            # wraps. Saying int32 here makes all three targets agree.
            wt: int32 = io.w[t]
            io.p_out.put(p + a * wt)
            io.a_out.put(a)


@spmw.unit
def tiled_vpu(io: TiledVpuIO):
    """A lane that reads an accumulator only when the program says to."""
    prog: int32[NPROG_T]
    for pc in range(NPROG_T):
        word: int32 = io.op_in.get()
        prog[pc] = word
        io.op_out.put(word)

    for m in range(MT):
        reg: int32[REGS]
        for step in range(NPROG_T):
            word2: int32 = prog[step]
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
def tpu_tiled(
    A: int8[MT * NTILE, KT],
    W: int8[KT, NT, NTILE],
    Bias: int32[NT],
    Prog: int32[NPROG_T],
    Y: int32[MT, NT],
):
    P = spmw.place(tiled_mac, on=tiled_mxu)
    V = spmw.place(tiled_vpu, on=tiled_chain)
    spmw.shard(W, into=P.w)  # cell (k, c) holds W[k, c, :], one weight per tile
    spmw.shard(Bias, into=V.b)
    # One streamed axis, so the activations arrive pre-tiled too: row
    # `m * NTILE + t` is what output row m needs from tile t, in the order the
    # cell's `m` outside `t` asks for them.
    spmw.stream_in(A, into=P.a_in, index=(..., P.rows))
    spmw.stream_in(0, into=P.p_in)
    spmw.link(P.p_out, to=V.z_in)
    spmw.stream_in(Prog, into=V.op_in, index=(...,))
    (lane,) = V.axes
    spmw.gather(Y, from_=V.y_out, index=(..., lane))


tpu_tiled.spmw_operands = {"Prog": TILED_RELU}


def _tiled_reference(A, W, Bias):
    """The whole reduction, in one numpy matmul -- the thing tiling must equal.

    The point of the test is that the engine's answer is a *single* matmul of
    depth NTILE*KT, so the reference does not tile at all: it undoes the
    per-cell weight layout and multiplies once.
    """
    flat_a = A.reshape(MT, NTILE * KT).astype(np.int32)  # (m, t*KT + k)
    # W[k, c, t] is the weight for reduction index t*KT + k
    flat_w = W.transpose(2, 0, 1).reshape(NTILE * KT, NT).astype(np.int32)
    return np.maximum(flat_a @ flat_w + Bias, 0) >> 4


def _tiled_operands(seed=2):
    rng = np.random.default_rng(seed)
    A = rng.integers(-16, 16, (MT * NTILE, KT)).astype(np.int8)
    W = rng.integers(-16, 16, (KT, NT, NTILE)).astype(np.int8)
    Bias = rng.integers(-64, 64, (NT,)).astype(np.int32)
    return A, W, Bias


def test_a_tiled_program_must_match_the_hardware():
    """One ACCZ per tile, or the lane and the array disagree about how many."""
    with pytest.raises(ValueError, match="ACCZ"):
        tiled_program((LOADI, 0, 0, 0), (ACCZ, 0), (STORE, 0))


def test_the_tiled_operands_exercise_the_program():
    A, W, Bias = _tiled_operands()
    out = _tiled_reference(A, W, Bias)
    assert np.count_nonzero(out) >= out.size // 3, out
    assert np.count_nonzero(out == 0) >= 1, out


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_the_tiled_engine_computes_the_deeper_matmul(target):
    """A reduction twice the array's depth, and the answer is the whole matmul."""
    A, W, Bias = _tiled_operands()
    Y = np.zeros((MT, NT), dtype=np.int32)
    spmw.build(tpu_tiled, target=target)(A, W, Bias, TILED_RELU, Y)
    np.testing.assert_array_equal(Y, _tiled_reference(A, W, Bias))


def test_a_block_memory_port_is_driven_whole():
    """The cosim driver must hand a cell *all* of its weights, not the first.

    `_memory_plan` collapsed each site's slice to its start, which is right for
    a scalar port and wrong for a block one: a cell holding NTILE weights got
    one, so every tile but the first reached the array as zero. It was silent --
    the array still ran, and only some outputs crossed the requantising shift
    differently, so cosim reported 5 wrong tokens out of 24 with no hint where
    they came from.

    This asks the plan the question directly: does the index it gives cover the
    whole block?
    """
    from allo.spmw import rtl

    graph = spmw.elaborate(tpu_tiled)
    plan = rtl.boundary_plan(graph)
    family = next(name for name in plan if name.endswith("_w_mem"))
    entry = plan[family]
    assert entry["tensor"] == "W"
    A, W, Bias = _tiled_operands()
    arrays = {"W": W}
    for channel in entry["channels"]:
        for index in channel:
            block = arrays[entry["tensor"]][index]
            assert block.shape == (NTILE,), (index, block.shape)
    del A, Bias


def test_every_cell_gets_its_own_weights():
    """And the blocks are the right ones, cell by cell."""
    from allo.spmw import rtl

    graph = spmw.elaborate(tpu_tiled)
    plan = rtl.boundary_plan(graph)
    entry = plan[next(n for n in plan if n.endswith("_w_mem"))]
    _A, W, _B = _tiled_operands()
    seen = []
    for channel in entry["channels"]:
        for index in channel:
            seen.append(W[index])
    flat = np.array(seen)
    assert flat.shape == (KT * NT, NTILE)
    # the union of every cell's block is the whole weight tensor, once
    np.testing.assert_array_equal(np.sort(flat.reshape(-1)), np.sort(W.reshape(-1)))
