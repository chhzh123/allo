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

# -- the VPU's extra opcode --------------------------------------------------
ACCN = 13  # [opcode:8 | dst:4 | 0:4 | n:16]: reg[dst] += the next n psums


def mxu_sweep(base, count):
    return (MSWEEP << 24) | (base << 16) | (count & 0xFFFF)


def mxu_pass(count):
    return (MPASS << 24) | (count & 0xFFFF)


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
    w = spmw.MemIn(int8[256])


def stage_engine(dim, kfile, outs, sweep, words=None, vprog_len=NPROG):
    """The fabric for one launch: `outs` result rows, each a `sweep`-tile sum.

    ``words`` is the MXU program length the buffer is built for; it defaults to
    one sweep word per output row. ``steps`` -- the activation stream -- is
    ``outs * sweep``.
    """
    words = outs if words is None else words
    steps = outs * sweep

    class CellIO(spmw.Interface):
        op_in = spmw.In(int32)
        op_out = spmw.Out(int32)
        a_in = spmw.In(int8)
        a_out = spmw.Out(int8)
        p_in = spmw.In(int32)
        p_out = spmw.Out(int32)
        w = spmw.MemIn(int8[kfile])

    mxu = spmw.Topology(
        CellIO,
        grid=(dim, dim),
        link=lambda i, j: {
            CellIO.a_out: spmw.to((i, j + 1), CellIO.a_in),
            CellIO.op_out: spmw.to((i, j + 1), CellIO.op_in),
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
        count: int32 = io.op_in.get()
        io.op_out.put(count)
        for _w in range(count):
            word: int32 = io.op_in.get()
            io.op_out.put(word)
            opcode: int32 = (word >> 24) & 255
            base: int32 = (word >> 16) & 255
            n: int32 = word & 65535
            # One word, many steps: the activation and the partial sum are
            # consumed every step whatever the opcode, so the streams stay in
            # lockstep with the lane that counts them.
            for t in range(n):
                a = io.a_in.get()
                p = io.p_in.get()
                io.a_out.put(a)
                if opcode == MSWEEP:
                    wt: int32 = io.w[base + t]
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
        W: int8[dim, dim, kfile],
        Bias: int32[dim, NB],
        MProg: int32[words + 1, dim],
        VProg: int32[vprog_len + 1],
        Y: int32[outs, dim],
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

    engine.spmw_parts = (mac, vpu, dim, kfile, outs, sweep, words, steps)
    return engine


# -- laying a GEMM stage out for the engine ----------------------------------


def stage_operands(X, Wmat, dim, kfile, shift, slabs=None, rng=None):
    """One launch of `X[R, K] @ Wmat[K, N]` for `slabs` 16-column slabs of N.

    Returns (A, W, Bias, MProg, VProg, expected). `A` is the activation stream
    the array reads: for each slab, each row, each K-tile, one 16-wide vector.
    `W` is the cell weight file: tile `s*T + t` is the (t, s) 16x16 block.
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
    mprog = mxu_program(words, dim)
    vprog = stage_vprog(T, shift, outs)
    bias = np.zeros((dim, NB), dtype=np.int32)
    full = X.astype(np.int32) @ Wmat.astype(np.int32)
    expected = np.zeros((outs, dim), dtype=np.int32)
    for s in range(slabs):
        expected[s * R : (s + 1) * R] = full[:, s * dim : (s + 1) * dim] >> shift
    return A, W, bias, mprog, vprog, expected


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
    assert proj[3].shape[0] - 1 == 512 and ffn2[3].shape[0] - 1 == 128  # words
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


def gpt_stage_of(size, kfile=256, rows=128, sweep=64, slabs=4):
    """The board shape: a `size` x `size` array, K=1024 in one sweep, four
    output slabs per launch -- 512 result rows, 32,768 activation steps."""
    return stage_engine(size, kfile, outs=slabs * rows, sweep=sweep)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
