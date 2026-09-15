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

import os

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


#: Depth of the mesh's links.  SPMW's default of two is what this uses, and
#: getting there took two measurements that interact.
#:
#: With the multiply inferred into a DSP the cell's iteration latency was 5,
#: and at that depth the assembled array ran at **half rate** -- `2S - 2`
#: cycles a tile with every loop reporting II=1 -- because a depth-2
#: `spmw_fifo` is a bare register slice whose `full_n` is a flop and the
#: producer could not re-offer inside that turnaround.  Depths 3, 4, 6 and 8
#: all fixed it and all gave byte-identical traces, so `LINK_DEPTH` was 4.
#:
#: Binding the multiply to fabric -- which the comparison against Gemmini
#: requires, since Gemmini spends no DSPs -- took the cell to an iteration
#: latency of **4**, and at 4 the depth-2 slice sustains full rate on its own.
#: So the deep links became a workaround for a problem that no longer exists,
#: and a costly one: at 16x16 they are 15,050 more lookup tables, 6,952 more
#: registers and 2 more cycles of first-tile latency for an identical
#: interval.  `LINK_DEEP` keeps that configuration measurable.
LINK_DEPTH = 2
LINK_DEEP = 4


def fixed_engine(dim, tiles, link_depth=LINK_DEPTH, weight_depth=None, reload_=False):
    """A `dim x dim` weight-stationary mesh with a hardwired requantise epilogue.

    `tiles` tiles live in the cell's weight file at once, so a launch is one
    weight load and then ``tiles * dim`` activation steps with no reload -- the
    same arrangement `test_spmw_tpu_micro` gives the programmable engine.
    """
    kfile = tiles
    if kfile % 4:
        raise ValueError(f"the packed weight file needs a multiple of 4, got {kfile}")
    kw = kfile // 4  # the file, in 32-bit words
    weight_depth = link_depth if weight_depth is None else weight_depth
    outs = tiles * dim  # one output row per activation step
    sbits = _log2(dim)  # step -> tile is a shift, not a divide

    class CellIO(spmw.Interface):
        """No `op` port: there is no instruction to carry.

        All three links are `link_depth` deep.  `a_in` and `p_in` carry a beat
        every cycle in the steady state, so the rate limit obviously binds on
        them; `w_in` moves only during the load, which is why it kept SPMW's
        default slice at first.  That was wrong, and measurably so: the load is
        *serial down the row*, every cell forwarding the words for the cells
        beyond it, so a half-rate weight link is paid `S x kw` times inside the
        first-tile latency.  Deepening it took 16x16 from 235 cycles to 213
        with the interval untouched, and the clock improved as well;
        `weight_depth` keeps the shallow one measurable.
        """

        a_in = spmw.In(int8, depth=link_depth)
        a_out = spmw.Out(int8)
        p_in = spmw.In(int32, depth=link_depth)
        p_out = spmw.Out(int32)
        w_in = spmw.In(int32, depth=weight_depth)
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

    # Two weight disciplines, as two separately decorated units rather than one
    # unit with a branch in it. A unit body is lowered from its AST, so a
    # Python-level `if` on a closure variable is *not* folded away -- it
    # becomes an `scf.if` whose condition is the captured integer, and the
    # verifier rejects that. The choice is made out here, before the decorator.
    #
    # `mac` below loads all `tiles` tiles once and keeps them packed: `8*tiles`
    # bits of state a cell, 128 at sixteen tiles, where Gemmini's PE holds one
    # tile double-buffered in 16 bits. That is most of why a cell is 364
    # flip-flops against a PE's 73, and it is a difference in what the designs
    # are rather than in how they are written.
    if reload_:

        @spmw.unit
        def mac(io: CellIO):
            """One tile of weight at a time, shifted in behind the arithmetic.

            The byte for the next tile arrives during this tile's steps and the
            bytes for the cells beyond are forwarded in the same loop, because
            a sibling loop with a runtime trip count would serialise behind the
            compute -- the shape that made the programmable engine's dispatch
            unpipelinable. This is what `d` does in Gemmini.
            """
            # How many of this tile's bytes belong to the cells beyond me. It
            # shrinks by one down the row, so it cannot be a compile-time
            # constant in a shared role; it arrives once, on the stream.
            fwd: int32 = io.w_in.get()
            io.w_out.put(fwd - 1)
            # Tile 0's weight up front: nothing has shifted it in yet. This is
            # Gemmini's warm-up pass, and it is one tile -- `dim` bytes down a
            # row -- not the whole file.
            cur: int32 = io.w_in.get()
            for _k in range(fwd):
                v0: int32 = io.w_in.get()
                io.w_out.put(v0)
            nxt: int32 = 0
            for t in range(tiles):
                for r in range(dim):
                    a = io.a_in.get()
                    p = io.p_in.get()
                    io.a_out.put(a)
                    io.p_out.put(p + a * cur)
                    if t + 1 < tiles:
                        if r <= fwd:
                            v: int32 = io.w_in.get()
                            if r == 0:
                                nxt = v
                            else:
                                io.w_out.put(v)
                cur = nxt

    else:

        @spmw.unit
        def mac(io: CellIO):
            # The weight file, filled once per launch. The count word is on the
            # weight stream rather than an instruction stream: a cell has to know
            # how many words belong to the cells beyond it, and that is the only
            # thing about this launch it does not know at compile time.
            #
            # The weights stay **packed**, four int8 to a 32-bit word, and are
            # unpacked in the step loop. Unpacking them here instead was tried,
            # because Gemmini's weight-stationary PE multiplies by a register
            # rather than by a file lookup, and it was a regression: the cell's
            # iteration latency stayed at 5 either way -- HLS was already hiding
            # the shift and mask -- while writing four int8 per word instead of one
            # int32 lengthened the load, which is serial down the row, and took
            # 8x8's first-tile latency from 112 cycles to 178. See the README.
            wf: int32[kw]
            n: int32 = io.w_in.get()
            io.w_out.put(n - kw)
            for i in range(kw):
                wf[i] = io.w_in.get()
            for _j in range(n - kw):
                fwd: int32 = io.w_in.get()
                io.w_out.put(fwd)

            # The step loop. Two things had to go from it, and the second was only
            # visible next to Gemmini's PE.
            #
            # The first is the instruction: the programmable cell's step loop is
            # nested inside a dispatch whose trip count is an instruction field, so
            # the dispatch cannot pipeline and the steps pay for it. Here there is
            # nothing above the steps.
            #
            # The second is the *weight fetch*. Gemmini's weight-stationary PE
            # multiplies by a register -- `mac_unit.io.in_b := c2`, with `c1 := d`
            # shifting the next tile's weight in behind it -- so its per-cycle work
            # is one multiply-add and its PE is one register deep. The first
            # version of this cell read `wf[idx >> 2]`, shifted, masked and
            # sign-extended on *every beat*, which is not weight-stationary at all;
            # HLS gave that loop five pipeline stages, and since a partial sum
            # crosses `dim` cells the five were paid `dim` times in the latency.
            # Hoisting the unpack to the tile boundary leaves `a * wt + p` against
            # a loop-invariant register, which is Gemmini's PE.
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

    wlen = 1 + tiles * dim if reload_ else dim * kw + 1

    @spmw.fabric
    def engine(
        A: int8[outs, dim],
        W: int32[wlen, dim],
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

    # Gemmini's `MxuVpu` routes with **zero** DSP blocks at every size: its
    # PE's multiply is Chisel arithmetic that Vivado maps to fabric. This
    # cell's identical `a * wt` is inferred into one DSP per element unless
    # bound, so without this the lookup-table columns are not measuring the
    # same thing -- one design has moved its arithmetic off the fabric being
    # counted. `SPMW_BIND_MUL=0` measures the DSP-inferred form instead.
    engine.spmw_bind_mul_fabric = os.environ.get("SPMW_BIND_MUL", "1") != "0"
    engine.spmw_parts = (mac, vpu, dim, kfile, outs)
    return engine


def reload_stream(size, B):
    """The weight stream a reloading row reads: a header, then a tile at a time.

    Row `k`'s stream carries, per tile, the bytes for cells `0 .. size-1` in
    order; cell `c` takes the first of what reaches it and passes the rest on.
    The header is `size - 1`, what cell 0 forwards, and it shrinks by one down
    the row. Sign-extended int32 tokens rather than packed bytes, because a
    cell wants one weight and not four.
    """
    tiles = B.shape[0]
    rows = [[size - 1] * size]
    for t in range(tiles):
        for c in range(size):
            rows.append([int(B[t][k][c]) for k in range(size)])
    return np.array(rows, dtype=np.int32)


def fixed_operands(size, A, B, bias, shift, reload_=False):
    """The three input tensors: activations, the weight stream, the constants.

    Two tensors fewer than the programmable engine, which is the point: the
    `MProg` and `VProg` streams have no counterpart here.
    """
    tiles = A.shape[0]
    kw = tiles // 4
    if reload_:
        consts0 = np.zeros((size, NBF), dtype=np.int32)
        consts0[:, 0] = bias
        consts0[:, 1] = shift
        return A.reshape(tiles * size, size), reload_stream(size, B), consts0
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


def micro_fixed_of(
    size, tiles=TILES, seed=0, link_depth=LINK_DEPTH, weight_depth=None, reload_=False
):
    """The fixed-function engine with the shared stimulus attached."""
    A, B, bias, shift, want = stimulus(size, tiles, seed)
    engine = fixed_engine(size, tiles, link_depth, weight_depth, reload_)
    names = ("A", "W", "Bias")
    engine.spmw_operands = dict(
        zip(names, fixed_operands(size, A, B, bias, shift, reload_))
    )
    engine.spmw_tokens_per_transform = size * size
    engine.spmw_micro = {
        "size": size,
        "tiles": tiles,
        "shift": int(shift),
        "fixed": True,
        "link_depth": link_depth,
        "weight_depth": link_depth if weight_depth is None else weight_depth,
        "reload": reload_,
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


# -- the Gemmini-matched weight discipline ------------------------------------


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_reloading_the_weights_computes_the_same_thing(target):
    """One tile of weight at a time, against the same golden result.

    The reloading cell is a different design -- it holds one tile where the
    file form holds `tiles` -- so it is checked, not assumed, against the same
    bytes both other systems are checked against.
    """
    size, tiles = 4, 4
    A, B, bias, shift, want = stimulus(size, tiles)
    tensors = fixed_operands(size, A, B, bias, shift, reload_=True)
    engine = fixed_engine(size, tiles, reload_=True)
    Y = np.zeros((tiles * size, size), dtype=np.int32)
    spmw.build(engine, target=target)(*tensors, Y)
    np.testing.assert_array_equal(Y, want.reshape(-1, size).astype(np.int32))


@pytest.mark.parametrize("size", [4, 8])
def test_both_weight_disciplines_agree(size):
    """They differ in what a cell stores and when it is told, not in the sum."""
    want = micro_fixed_of(size).spmw_micro["expected"]
    for reload_ in (False, True):
        engine = micro_fixed_of(size, reload_=reload_)
        arrays = [engine.spmw_operands[n] for n in ("A", "W", "Bias")]
        Y = np.zeros(want.shape, dtype=np.int32)
        spmw.build(engine, target="ref")(*arrays, Y)
        np.testing.assert_array_equal(Y, want, err_msg=f"reload_={reload_}")


def test_the_reloading_stream_carries_every_weight_once():
    """Same weights, different transport: the check that the stream is right."""
    size, tiles = 8, 4
    A, B, bias, shift, _ = stimulus(size, tiles)
    _, w_file, _ = fixed_operands(size, A, B, bias, shift, reload_=False)
    _, w_reload, _ = fixed_operands(size, A, B, bias, shift, reload_=True)
    assert w_file.shape == (size * (tiles // 4) + 1, size)
    assert w_reload.shape == (1 + tiles * size, size)
    for k in range(size):
        got = sorted(int(v) for v in w_reload[1:, k])
        wants = sorted(int(B[t][k][c]) for t in range(tiles) for c in range(size))
        assert got == wants, k
