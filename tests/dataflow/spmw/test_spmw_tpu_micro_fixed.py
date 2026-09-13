# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The E3 microbenchmark with SPMW's programmability taken out.

`test_spmw_tpu_micro` runs the workload on `gpt_stage_v1.stage_engine`, the
netlist that ran a GPT-2 block on the U280: a matrix cell that fetches an
instruction word and a vector lane that fetches a program.  Measured against
Gemmini's fixed-function `MxuVpu` it loses 27x to 36x, and the reports say why
in one line -- both dispatch loops are ``Pipelined no``, because each holds a
nested loop whose trip count is an instruction field.

**This is the same workload on a fixed-function SPMW datapath.**  The mesh and
the epilogue are the same arithmetic in the same order over the same operands;
what is gone is the ability to be told to do anything else:

===========================  ==================================  ================
                             programmable (`tpumicro`)           fixed (this)
===========================  ==================================  ================
cell control                 fetch, decode, ``MSWEEP``/``MPASS`` one flat loop
lane control                 16-word program, 14-opcode decode    straight line
the clip                     five instructions out of MAX/SUB     one comparison
``MUL`` opcode               four DSPs a lane, never issued       not present
instruction chain            an ``op`` link across the whole mesh not present
===========================  ==================================  ================

Everything else is deliberately held constant, so the pair is an ablation and
not two designs: the same 32-bit packed weight file resident for all sixteen
tiles, the same one-psum-per-output dataflow, the same operands from
`test_spmw_tpu_micro.stimulus`, the same golden result.

**The shift stays a runtime input.**  It arrives in the lane's constant memory
next to the bias rather than being baked in as a Python constant, because
Gemmini's `AccumulatorScale` takes its scale per command -- a folded constant
would turn SPMW's barrel shifter into wiring and win the area column by
answering a different question.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int8, int32

from gpt_stage_v1 import file_to_stream
from test_spmw_tpu_micro import INT8_MAX, TILES, stimulus

#: Per-lane constants: the bias, and the requantisation shift.
NBF = 2


def _log2(n):
    bits = n.bit_length() - 1
    if 1 << bits != n:
        raise ValueError(f"the array side must be a power of two, got {n}")
    return bits


#: Depth of the mesh's data links.  Two -- SPMW's default, a register slice --
#: is a *credit* limit and not just a buffer: a producer writing into a slice
#: can only run as fast as the slot it filled comes back, and across the split
#: fabric that round trip is four cycles, so two slots sustain one beat every
#: two cycles no matter what the units achieve.  Both units here run at II=1
#: and the assembled array still measured `2S - 2` cycles a tile, which is
#: exactly half rate; at eight the credit stops binding and the array runs at
#: the II its loops report.  `LINK_SLICE` keeps the default measurable.
LINK_DEPTH = 8
LINK_SLICE = 2


def fixed_engine(dim, tiles, link_depth=LINK_DEPTH):
    """A `dim x dim` weight-stationary mesh with a hardwired requantise epilogue.

    `tiles` tiles live in the cell's weight file at once, so a launch is one
    weight load and then ``tiles * dim`` activation steps with no reload -- the
    same arrangement `test_spmw_tpu_micro` gives the programmable engine.
    """
    kfile = tiles
    if kfile % 4:
        raise ValueError(f"the packed weight file needs a multiple of 4, got {kfile}")
    kw = kfile // 4  # the file, in 32-bit words
    outs = tiles * dim  # one output row per activation step
    sbits = _log2(dim)  # step -> tile is a shift, not a divide

    class CellIO(spmw.Interface):
        """No `op` port: there is no instruction to carry.

        `a_in` and `p_in` carry a beat every cycle in the steady state and are
        the two the credit limit binds on.  `w_in` runs once per launch, so it
        keeps the default slice -- a deeper weight link would buy nothing and
        cost a LUT-RAM per cell.
        """

        a_in = spmw.In(int8, depth=link_depth)
        a_out = spmw.Out(int8)
        p_in = spmw.In(int32, depth=link_depth)
        p_out = spmw.Out(int32)
        w_in = spmw.In(int32)
        w_out = spmw.Out(int32)

    class LaneIO(spmw.Interface):
        """No `op` port either, so the lanes are not even a chain."""

        z_in = spmw.In(int32, depth=link_depth)
        y_out = spmw.Out(int32)
        b = spmw.MemIn(int32[NBF])

    mxu = spmw.Topology(
        CellIO,
        grid=(dim, dim),
        link=lambda i, j: {
            CellIO.a_out: spmw.to((i, j + 1), CellIO.a_in),
            CellIO.w_out: spmw.to((i, j + 1), CellIO.w_in),
            CellIO.p_out: spmw.to((i + 1, j), CellIO.p_in),
        },
    )
    lanes = spmw.Grid((dim,))

    @spmw.unit
    def mac(io: CellIO):
        # The weight file, filled once per launch. The count word is on the
        # weight stream rather than an instruction stream: a cell has to know
        # how many words belong to the cells beyond it, and that is the only
        # thing about this launch it does not know at compile time.
        wf: int32[kw]
        n: int32 = io.w_in.get()
        io.w_out.put(n - kw)
        for i in range(kw):
            wf[i] = io.w_in.get()
        for _j in range(n - kw):
            fwd: int32 = io.w_in.get()
            io.w_out.put(fwd)

        # One flat loop with a compile-time trip count, and no branch on an
        # opcode inside it. This is the whole difference: the programmable
        # cell's step loop is nested inside a dispatch whose trip count is an
        # instruction field, so the dispatch cannot pipeline and the steps pay
        # for it; here there is nothing above the steps.
        for r in range(outs):
            a = io.a_in.get()
            p = io.p_in.get()
            io.a_out.put(a)
            idx: int32 = r >> sbits  # dim steps per tile
            packed: int32 = wf[idx >> 2]
            byte: int32 = (packed >> ((idx & 3) * 8)) & 255
            wt: int32 = (byte ^ 128) - 128
            io.p_out.put(p + a * wt)

    @spmw.unit
    def vpu(io: LaneIO):
        bias: int32 = io.b[0]
        sh: int32 = io.b[1]
        for _m in range(outs):
            z: int32 = io.z_in.get()
            acc: int32 = bias + z
            if acc < 0:  # ReLU
                acc = 0
            acc = acc >> sh  # requantise, by a runtime amount
            if acc > INT8_MAX:  # the clip, as a comparison rather than a program
                acc = INT8_MAX
            io.y_out.put(acc)

    @spmw.fabric
    def engine(
        A: int8[outs, dim],
        W: int32[dim * kw + 1, dim],
        Bias: int32[dim, NBF],
        Y: int32[outs, dim],
    ):
        P = spmw.place(mac, on=mxu)
        V = spmw.place(vpu, on=lanes)
        spmw.shard(Bias, into=V.b)
        spmw.stream_in(A, into=P.a_in, index=(..., P.rows))
        spmw.stream_in(W, into=P.w_in, index=(..., P.rows))
        spmw.stream_in(0, into=P.p_in)
        spmw.link(P.p_out, to=V.z_in)
        (lane,) = V.axes
        spmw.gather(Y, from_=V.y_out, index=(..., lane))

    engine.spmw_parts = (mac, vpu, dim, kfile, outs)
    return engine


def fixed_operands(size, A, B, bias, shift):
    """The three input tensors: activations, the weight stream, the constants.

    Two tensors fewer than the programmable engine, which is the point: the
    `MProg` and `VProg` streams have no counterpart here.
    """
    tiles = A.shape[0]
    kw = tiles // 4
    weights = np.zeros((size, size, tiles), dtype=np.int8)
    for t in range(tiles):
        weights[:, :, t] = B[t]
    stream = file_to_stream(weights)  # [size * kw, size]
    header = np.full((1, size), size * kw, dtype=np.int32)
    consts = np.zeros((size, NBF), dtype=np.int32)
    consts[:, 0] = bias
    consts[:, 1] = shift
    return (
        A.reshape(tiles * size, size),
        np.concatenate([header, stream], axis=0),
        consts,
    )


def micro_fixed_of(size, tiles=TILES, seed=0, link_depth=LINK_DEPTH):
    """The fixed-function engine with the shared stimulus attached."""
    A, B, bias, shift, want = stimulus(size, tiles, seed)
    engine = fixed_engine(size, tiles, link_depth)
    names = ("A", "W", "Bias")
    engine.spmw_operands = dict(zip(names, fixed_operands(size, A, B, bias, shift)))
    engine.spmw_tokens_per_transform = size * size
    engine.spmw_micro = {
        "size": size,
        "tiles": tiles,
        "shift": int(shift),
        "fixed": True,
        "link_depth": link_depth,
        "expected": want.reshape(tiles * size, size).astype(np.int32),
    }
    return engine


# -- tests --------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_four_tiles_bit_exact(target):
    """A 4x4 array, four tiles, against the shared golden."""
    size, tiles = 4, 4
    A, B, bias, shift, want = stimulus(size, tiles)
    tensors = fixed_operands(size, A, B, bias, shift)
    engine = fixed_engine(size, tiles)
    Y = np.zeros((tiles * size, size), dtype=np.int32)
    spmw.build(engine, target=target)(*tensors, Y)
    np.testing.assert_array_equal(Y, want.reshape(-1, size).astype(np.int32))


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_eight_by_eight(target):
    """A second size through the same two roles."""
    size, tiles = 8, 4
    A, B, bias, shift, want = stimulus(size, tiles)
    tensors = fixed_operands(size, A, B, bias, shift)
    engine = fixed_engine(size, tiles)
    Y = np.zeros((tiles * size, size), dtype=np.int32)
    spmw.build(engine, target=target)(*tensors, Y)
    np.testing.assert_array_equal(Y, want.reshape(-1, size).astype(np.int32))


@pytest.mark.parametrize("size", [4, 8, 16])
def test_the_measured_configuration_matches_the_golden(size):
    """What the cosimulation runs must compute the shared golden result.

    The array cosimulation checks the RTL against ``target="ref"`` on whatever
    the fabric carries, not against the golden, so without this the fixed
    engine could pass its own cosim while computing something Gemmini is not
    computing -- and the comparison would be two columns again.
    """
    engine = micro_fixed_of(size)
    want = engine.spmw_micro["expected"]
    arrays = [engine.spmw_operands[n] for n in ("A", "W", "Bias")]
    Y = np.zeros(want.shape, dtype=np.int32)
    spmw.build(engine, target="ref")(*arrays, Y)
    np.testing.assert_array_equal(Y, want)


def test_it_is_the_same_workload_as_the_programmable_engine():
    """Both engines must be handed the same operands, or the ablation is not one.

    Same activations, same weights, same bias, same shift, same expected
    result -- the fixed engine differs in its control, not in its arithmetic.
    """
    from test_spmw_tpu_micro import micro_of

    prog = micro_of(8)
    fixed = micro_fixed_of(8)
    np.testing.assert_array_equal(
        prog.spmw_micro["expected"], fixed.spmw_micro["expected"]
    )
    assert prog.spmw_micro["shift"] == fixed.spmw_micro["shift"]
    np.testing.assert_array_equal(prog.spmw_operands["A"], fixed.spmw_operands["A"])
    # The weight stream is the programmable engine's with the count word moved
    # off the instruction stream and onto the weight stream.
    np.testing.assert_array_equal(
        prog.spmw_operands["W"], fixed.spmw_operands["W"][1:]
    )
    np.testing.assert_array_equal(
        prog.spmw_operands["Bias"][:, 0], fixed.spmw_operands["Bias"][:, 0]
    )


def test_the_shift_is_an_input_and_not_a_constant():
    """Fold it and SPMW wins area by answering a different question.

    Gemmini's `AccumulatorScale` takes its scale per command, so a barrel
    shifter is what a fair comparison costs.
    """
    engine = micro_fixed_of(4)
    consts = engine.spmw_operands["Bias"]
    assert consts.shape[1] == NBF
    assert (consts[:, 1] == engine.spmw_micro["shift"]).all()
