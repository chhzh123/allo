# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A programmable mini-TPU on the 256-unit array, dispatching a row a cycle.

E3's stage engine was programmable and slow. Its cells ran one loop per MXU
instruction word and its vector lane ran a ten-instruction micro-program per
output row, so it spent 41 cycles a row: 27 to 36 times Gemmini's interval.
Everything after it in E3 and E9 was fixed-function, with the workload compiled
into the array.

This engine is the 256-unit array of `test_spmw_tpu_micro_lean` and
`test_spmw_tpu_micro_blocked` -- Gemmini's weight shift, one multiply-add a
cell, bare-register links -- driven by a program instead of a compiled
workload. Its dispatch is a three-stage pipeline, a row a cycle end to end:

* **Fetch and decode: one sequencer.** `seq` reads a 64-bit instruction per
  GEMM and issues one 12-bit micro-op per output row. It is one flat loop: the
  block counters move on each block's last row, and the next instruction is
  read and decoded in the first row of the GEMM it describes, so an
  instruction boundary costs no cycle. The whole array has one set of
  counters.
* **Distribute: a tap a lane.** The micro-ops pass along a row of taps, one a
  lane, at the array's own skew of a lane a cycle. A tap is one register stage
  that hands its lane a copy; it is checked to be a single stage, as a cell
  is, so the control wave cannot fall behind the data it describes.
* **Execute: the lanes.** A lane holds no counter and decodes nothing. It
  reads a micro-op with each partial sum and does what the bits say.

The cells run no program at all. An activation token is the int8 value and two
framing bits the stream itself carries, as a bus carries TLAST: `SWAP` on
every ``S``-th beat, where a cell hands the weight it has shifted in to its
multiplier, and `LAST` on the launch's final beat. A launch opens with ``S``
beats that shift the first block's weights in; a cell passes no partial sum
until its first `SWAP`, so the lanes never see them. A cell is two weight
registers, a multiply-add and two flags, with no counter.

**An instruction** is one GEMM, 64 bits:

    (REP - 1) << 32 | (KB - 1) << 16 | FINAL << 8 | BIAS | RAW | RELU | SHIFT

``REP`` accumulations, each the sum of ``KB`` K blocks of ``S`` rows: a GEMM
over ``T`` row tiles and ``NB`` column blocks is ``REP = T * NB``. `BIAS`
seeds each accumulation with the lane's bias, `RELU` clamps the sum at zero
before the shift, and `RAW` emits the int32 sum untouched; otherwise the sum
is shifted right by ``SHIFT`` and clipped to int8 -- E3's epilogue with
`BIAS | RELU`, LLaMA's projections with neither. `FINAL` marks the program's
last instruction. Biases arrive on a stream of their own, one token per lane
per accumulation.

**A micro-op** is the instruction's low eight bits and four the sequencer
adds: `FIRST` on the rows of an accumulation's first K block, `EMIT` on those
of its last, `LOAD` on the row whose lane must pop a bias, and `END` on the
launch's last row.

The hardware never sees a workload's shape. One build per array size runs every
program, which `test_one_engine_for_every_program` checks by compiling two
workloads and comparing every role's code and the fabric.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int8, int16, int32, int64, uint1, uint8, uint16

from test_spmw_tpu_micro import INT8_MAX, TILES, stimulus_mkn
from test_spmw_tpu_micro_lean import _generated
from test_spmw_llama_ffn import TOKENS, SLICE, llama_stimulus

#: An instruction's epilogue flags, above its five-bit shift, and its end mark.
RELU, RAW, BIAS, FINAL = 1 << 5, 1 << 6, 1 << 7, 1 << 8
#: What the sequencer adds to make a micro-op.
FIRST, EMIT, LOAD, END = 1 << 8, 1 << 9, 1 << 10, 1 << 11
#: An activation token's framing, above the int8 value.
SWAP, LAST = 1 << 8, 1 << 9


def _lane_source(S):
    """The lane: a delay line, an adder and the epilogue, told what to do.

    ``r0 .. r{S-1}`` is the delay line: a block is ``S`` rows, so the value
    pushed ``S`` rows ago is the same row's sum one K block back. Every other
    decision is a bit of the row's micro-op.
    """
    lines = [
        "def make(INT8_MAX, IO):",
        "    def lane(io: IO):",
        "        bias: int32 = 0",
        "        go: uint1 = 1",
    ]
    lines += [f"        r{i}: int32 = 0" for i in range(S)]
    lines += [
        "        while go != 0:",
        "            c: int16 = io.c_in.get()",
        "            z: int32 = io.z_in.get()",
        "            if ((c >> 10) & 1) != 0:",  # LOAD
        "                bias = io.b_in.get()",
        "            s: int32 = r0",
        "            if ((c >> 8) & 1) != 0:",  # FIRST
        "                s = 0",
        "                if ((c >> 7) & 1) != 0:",  # BIAS
        "                    s = bias",
        "            v: int32 = s + z",
    ]
    lines += [f"            r{i} = r{i + 1}" for i in range(S - 1)]
    lines += [
        f"            r{S - 1} = v",
        "            if ((c >> 9) & 1) != 0:",  # EMIT
        "                y: int32 = v",
        "                if ((c >> 6) & 1) == 0:",  # not RAW
        "                    if ((c >> 5) & 1) != 0:",  # RELU
        "                        if y < 0:",
        "                            y = 0",
        "                    y = y >> (c & 31)",
        "                    if y < -128:",
        "                        y = -128",
        "                    if y > INT8_MAX:",
        "                        y = INT8_MAX",
        "                io.y_out.put(y)",
        "            if ((c >> 11) & 1) != 0:",  # END
        "                go = 0",
        "    return lane",
    ]
    return "\n".join(lines) + "\n"


def ptpu_engine(S, na, nw, nprog, nbias, ny):
    """The programmable ``S x S`` engine, with edge tensors of the given lengths.

    The lengths size the harness's streams, not the hardware: every unit is
    the same whatever program runs.
    """
    lastrow = S - 1

    class PEIO(spmw.Interface):
        """The lean cell's links, every one a bare register."""

        a_in = spmw.In(int16, depth=0)
        a_out = spmw.Out(int16, depth=0)
        p_in = spmw.In(int32, depth=0)
        p_out = spmw.Out(int32, depth=0)
        w_in = spmw.In(int8, depth=0)
        w_out = spmw.Out(int8, depth=0)

    class SeqIO(spmw.Interface):
        op_in = spmw.In(int64)  # the program, an instruction a GEMM
        u_out = spmw.Out(int16)  # the micro-ops, one a row

    class TapIO(spmw.Interface):
        u_in = spmw.In(int16)
        u_out = spmw.Out(int16)
        c_out = spmw.Out(int16)  # this lane's copy

    class LaneIO(spmw.Interface):
        c_in = spmw.In(int16)
        z_in = spmw.In(int32, depth=2)
        b_in = spmw.In(int32)  # this lane's biases, one per accumulation
        y_out = spmw.Out(int32)

    mxu = spmw.Topology(
        PEIO,
        grid=(S, S),
        link=lambda i, j: {
            PEIO.a_out: spmw.to((i, j + 1), PEIO.a_in),
            PEIO.w_out: spmw.to((i, j + 1), PEIO.w_in),
            PEIO.p_out: spmw.to((i + 1, j), PEIO.p_in),
        },
    )
    alone = spmw.Topology(SeqIO, grid=(1,), link=lambda i: {})
    row = spmw.Topology(
        TapIO,
        grid=(S,),
        link=lambda i: {TapIO.u_out: spmw.to((i + 1,), TapIO.u_in)},
    )
    lanes = spmw.Topology(LaneIO, grid=(S,), link=lambda i: {})

    @spmw.unit
    def pe(io: PEIO):
        # `lean_engine`'s one-loop cell, run by its tokens and counting
        # nothing. Until the first `SWAP` it only shifts weights along; after
        # it, every beat is a multiply-add. The loop ends on the stream's last
        # beat.
        nxt: int8 = 0
        cur: int8 = 0
        live: uint1 = 0
        go: uint1 = 1
        while go != 0:
            t: int16 = io.a_in.get()
            io.a_out.put(t)
            if live != 0:
                a: int8 = ((t & 255) ^ 128) - 128
                p = io.p_in.get()
                io.p_out.put(p + a * cur)
            io.w_out.put(nxt)
            nxt = io.w_in.get()
            if ((t >> 8) & 1) != 0:
                cur = nxt
                live = 1
            if ((t >> 9) & 1) != 0:
                go = 0

    @spmw.unit
    def seq(io: SeqIO):
        # One iteration a row. `kc` counts an accumulation's K blocks down and
        # `rc` the GEMM's accumulations; both move on a block's last row.
        # `klast` and `rlast` say a counter is on its last, and are kept as
        # flags so that no row waits on a comparison. The GEMM's last row asks
        # for the next instruction, which the next row reads and decodes as it
        # issues its own micro-op.
        cfg: int16 = 0
        final: uint1 = 0
        kn: uint16 = 0
        kc: uint16 = 0
        rc: int32 = 0
        first: uint1 = 1
        kone: uint1 = 0
        klast: uint1 = 0
        rlast: uint1 = 0
        need: uint1 = 1
        ph: uint8 = 0
        go: uint1 = 1
        while go != 0:
            if need != 0:
                ins: int64 = io.op_in.get()
                cfg = ins & 255
                final = (ins >> 8) & 1
                kn = (ins >> 16) & 65535
                kc = kn
                rc = ins >> 32
                kone = 0
                if kn == 0:
                    kone = 1
                klast = kone
                rlast = 0
                if rc == 0:
                    rlast = 1
                need = 0
            u: int16 = cfg
            if first != 0:
                u = u | 256
                if ph == 0:
                    if ((cfg >> 7) & 1) != 0:
                        u = u | 1024
            if klast != 0:
                u = u | 512
            if ph == lastrow:
                ph = 0
                if klast != 0:
                    kc = kn
                    first = 1
                    klast = kone
                    if rlast != 0:
                        need = 1
                        if final != 0:
                            u = u | 2048
                            go = 0
                    else:
                        if rc == 1:
                            rlast = 1
                        rc = rc - 1
                else:
                    first = 0
                    if kc == 1:
                        klast = 1
                    kc = kc - 1
            else:
                ph = ph + 1
            io.u_out.put(u)

    @spmw.unit
    def tap(io: TapIO):
        go: uint1 = 1
        while go != 0:
            u: int16 = io.u_in.get()
            io.u_out.put(u)
            io.c_out.put(u)
            if ((u >> 11) & 1) != 0:
                go = 0

    lane = _generated(f"plane{S}", _lane_source(S), INT8_MAX=INT8_MAX, IO=LaneIO)

    @spmw.fabric
    def engine(
        A: int16[na, S],
        W: int8[nw, S],
        Prog: int64[nprog],
        Bias: int32[nbias, S],
        Y: int32[ny, S],
    ):
        P = spmw.place(pe, on=mxu)
        spmw.pipeline(P, ii=1, combinational=True)
        Q = spmw.place(seq, on=alone)
        spmw.pipeline(Q, ii=1, combinational=True)
        T = spmw.place(tap, on=row)
        spmw.pipeline(T, ii=1, combinational=True)
        V = spmw.place(lane, on=lanes)
        spmw.pipeline(V, ii=1)
        (col,) = V.axes
        spmw.stream_in(A, into=P.a_in, index=(..., P.rows))
        spmw.stream_in(W, into=P.w_in, index=(..., P.rows))
        spmw.stream_in(0, into=P.p_in)
        spmw.link(P.p_out, to=V.z_in)
        spmw.stream_in(Prog, into=Q.op_in, index=(...,))
        spmw.link(Q.u_out, to=T.u_in)
        spmw.link(T.c_out, to=V.c_in)
        spmw.stream_in(Bias, into=V.b_in, index=(..., col))
        spmw.gather(Y, from_=V.y_out, index=(..., col))

    engine.spmw_bind_mul_fabric = True
    return engine


# -- writing programs ---------------------------------------------------------


class Gemm:
    """One GEMM instruction and the data it consumes.

    ``tiles`` is one ``(X, W)`` per row tile: ``X`` is the tile's ``S``
    activation rows, ``[S, K]``, and ``W`` its weights, ``[K, N]`` -- the same
    matrix for a layer's tokens, one per tile for E3's microbenchmark.
    """

    def __init__(self, tiles, flags=0, shift=0, bias=None):
        self.tiles = tiles
        self.flags = flags
        self.shift = shift
        self.bias = bias

    def golden(self, S):
        """The lanes' output for this GEMM, in the order they emit it."""
        outs = []
        for X, W in self.tiles:
            acc = X.astype(np.int64) @ W.astype(np.int64)
            if self.flags & BIAS:
                acc = acc + self.bias
            if not self.flags & RAW:
                if self.flags & RELU:
                    acc = np.maximum(acc, 0)
                acc = np.clip(acc >> self.shift, -128, INT8_MAX)
            for nb in range(W.shape[1] // S):
                outs.append(acc[:, nb * S : (nb + 1) * S])
        return outs


def assemble(S, gemms):
    """A launch: the streams, the program and the golden.

    Returns ``(A, W, Prog, Bias, Y)``. Blocks are consumed tile by tile, then
    column block, then K block, each for ``S`` rows. The activation stream
    opens with ``S`` beats of nothing, marks every block's last beat `SWAP`
    and its own last `LAST`. The weight stream carries each block one block
    ahead of its rows -- the first under the opening beats -- columns
    ``S-1 .. 0``, so after ``S`` shifts cell ``j`` holds column ``j``. The
    bias stream has a row per accumulation of each `BIAS` GEMM, lane ``j``
    reading column ``j``.
    """
    rows, blocks, outs, prog, biases = [], [], [], [], []
    for g in gemms:
        K, N = g.tiles[0][1].shape
        KB, NB = K // S, N // S
        rep = len(g.tiles) * NB
        if KB > 1 << 16 or rep >= 1 << 31 or g.shift > 31:
            raise ValueError("a GEMM field does not fit its instruction")
        for X, W in g.tiles:
            for nb in range(NB):
                if g.flags & BIAS:
                    biases.append(g.bias[nb * S : (nb + 1) * S])
                for kb in range(KB):
                    rows.append(X[:, kb * S : (kb + 1) * S])
                    blocks.append(W[kb * S : (kb + 1) * S, nb * S : (nb + 1) * S])
        prog.append(((rep - 1) << 32) | ((KB - 1) << 16) | g.flags | g.shift)
        outs += g.golden(S)
    prog[-1] |= FINAL
    data = np.concatenate(rows).astype(np.int16) & 255
    A = np.concatenate([np.zeros((S, S), dtype=np.int16), data])
    A[S - 1 :: S, :] |= SWAP
    A[-1, :] |= LAST
    W = np.zeros(((len(blocks) + 1) * S, S), dtype=np.int8)
    for q, blk in enumerate(blocks):
        W[q * S : (q + 1) * S, :] = blk[:, ::-1].T
    bias = np.array(biases or [np.zeros(S)], dtype=np.int32)
    return (
        A,
        W,
        np.array(prog, dtype=np.int64),
        bias,
        np.concatenate(outs).astype(np.int32),
    )


def ptpu_of(S, gemms):
    """The engine with one launch's operands and golden attached."""
    A, W, prog, bias, Y = assemble(S, gemms)
    engine = ptpu_engine(S, len(A), len(W), len(prog), len(bias), len(Y))
    engine.spmw_operands = {"A": A, "W": W, "Prog": prog, "Bias": bias}
    engine.spmw_expected = {"Y": Y}
    engine.spmw_cosim_cycles = len(W) + 64 * S + 10000
    engine.spmw_ptpu = {"size": S, "rows": len(W) - S, "expected": Y}
    return engine


# -- the workloads, as programs -----------------------------------------------


def micro_gemms(S, tiles=TILES, seed=0):
    """E3's microbenchmark: `tiles` 16x16x16 tiles, each its own weights."""
    A, B, bias, shift, _ = stimulus_mkn(tiles, 16, 16, 16, seed)
    pieces = [
        (A[t][rt * S : (rt + 1) * S, :], B[t])
        for t in range(tiles)
        for rt in range(16 // S)
    ]
    return [Gemm(pieces, RELU | BIAS, shift, bias)]


def llama_gemms(S, L=TOKENS, K=2048, n=SLICE):
    """The gate-and-up slice of `test_spmw_llama_ffn`: requantised, no bias."""
    x, w, shift, _shift2, _gu, _h = llama_stimulus(L, K, n)
    return [Gemm([(x[t * S : (t + 1) * S, :], w) for t in range(L // S)], 0, shift)]


def raw_gemm(S, K, N, seed):
    """One tile of a GEMM whose int32 sums come out untouched."""
    rng = np.random.default_rng(seed)
    X = rng.integers(-128, 128, (S, K)).astype(np.int8)
    return Gemm([(X, rng.integers(-128, 128, (K, N)).astype(np.int8))], RAW)


def mixed_gemms(S):
    """Five GEMMs of four shapes back to back, down to one block of ``S`` rows.

    What a single-GEMM launch cannot show: that an instruction boundary costs
    no cycle, whatever is on either side of it.
    """
    return (
        micro_gemms(S, tiles=2)
        + [raw_gemm(S, S, S, 5)]
        + llama_gemms(S, L=16, K=48, n=8)
        + [raw_gemm(S, 3 * S, 2 * S, 6), raw_gemm(S, S, S, 7)]
    )


def ptpu_workload(S, name):
    """One of the workloads E9 runs, as a launch of this engine."""
    gemms = {
        "micro": lambda: micro_gemms(S),
        "llama": lambda: llama_gemms(S),
        "dsv4": lambda: llama_gemms(S, K=7168),
        "mixed": lambda: mixed_gemms(S),
    }[name]()
    return ptpu_of(S, gemms)


# -- tests --------------------------------------------------------------------


def _run(engine, target):
    want = engine.spmw_ptpu["expected"]
    Y = np.zeros(want.shape, dtype=np.int32)
    ops = [engine.spmw_operands[k] for k in ("A", "W", "Prog", "Bias")]
    spmw.build(engine, target=target)(*ops, Y)
    np.testing.assert_array_equal(Y, want)


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_the_microbenchmark_runs_as_a_program(target):
    _run(ptpu_of(4, micro_gemms(4, tiles=2)), target)


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_a_layer_runs_as_a_program(target):
    _run(ptpu_of(4, llama_gemms(4, L=16, K=32, n=8)), target)


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_several_gemms_run_back_to_back(target):
    """Five instructions, four shapes, three epilogues, one launch."""
    _run(ptpu_of(4, mixed_gemms(4)), target)


def test_a_program_is_an_instruction_a_gemm():
    """However long the launch, its program is one word per GEMM."""
    _A, W, prog, _bias, _Y = assemble(4, mixed_gemms(4))
    assert len(prog) == 5 and len(W) > 100 * len(prog)
    assert [int(p) >> 8 & 1 for p in prog] == [0, 0, 0, 0, 1]
    # The layer GEMM: 4 tiles of 4 rows by 4 column blocks (gate and up, 8
    # columns each), 12 K blocks each.
    assert int(prog[2]) >> 32 == 4 * 4 - 1
    assert int(prog[2]) >> 16 & 65535 == 12 - 1


def hardware(engine):
    """Everything the engine compiles to: each role's HLS C++, and the fabric."""
    # pylint: disable=import-outside-toplevel
    from allo.spmw import rtl
    from allo.spmw.role_ip import UnitEmitter, build_unit

    graph = spmw.elaborate(engine)
    emitter = UnitEmitter(graph)
    code = {}
    for placement in emitter.placements():
        for order in range(len(emitter.classes(placement))):
            built = build_unit(graph, placement, order, target="vhls")
            code[emitter.role_name(placement, order)] = str(built.hls_code)
    code["spmw_top.sv"] = rtl.StructuralEmitter(graph).fabric()
    return code


def test_one_engine_for_every_program():
    """Two workloads of different shapes and epilogues compile to one engine."""
    a = ptpu_of(4, micro_gemms(4, tiles=1))
    b = ptpu_of(4, llama_gemms(4, L=16, K=48, n=8))
    assert a.spmw_ptpu["rows"] != b.spmw_ptpu["rows"]
    ha, hb = hardware(a), hardware(b)
    assert sorted(ha) == sorted(hb)
    for role, code in ha.items():
        assert code == hb[role], f"{role} differs between the two programs"
