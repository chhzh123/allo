# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The E3 microbenchmark with Gemmini's processing element, written in SPMW.

`test_spmw_tpu_micro_fixed` is 511 flip-flops a cell against Gemmini's 73.
Routed at 16x16, 362 of them are the cell body and ~140 its three links, and
two things in how that cell is written put the body there.

**The weight.**  The fixed cell keeps the whole 16-tile file resident, 128
bits, loaded by a counted prologue and forwarded by a second counted loop, and
unpacks a byte out of it with a barrel shifter on every beat.  Gemmini's PE
holds one int8 and a second one behind it: the next tile's weight shifts down
the row *through* the cells while this tile computes, and a flag swaps them.
Here the same: every cell passes its ``nxt`` on and takes a new one each beat,
so after ``S`` beats cell ``j`` holds the ``S - 1 - j``-th token -- the row's
weights sent in reverse order.  No file, no count, no position: every cell is
the same, and the weight is a register.

**The pipeline.**  At a 3.33 ns target Vitis charges each FIFO access 1.21 ns,
so read, multiply, add and write -- 4.64 ns in its model -- cannot share a
stage, and it registers ``a``, ``p``, the product and the sum *on top of* the
link registers that already make the array a pipeline.  Gemmini's PE is one
multiply-add between two registers (``tile_latency = 0``);
``spmw.pipeline(P, ii=1, combinational=True)`` asks for the same.

**The block.**  After those two the links are the largest item, 58% of the
array's registers.  `fused_engine` puts `f x f` cells in one unit, which has
`f` links of each kind where its cells had `f^2`; at `f = 4` the array is
smaller than Gemmini's.

The epilogue lane and the operands are the fixed engine's; the weight stream
is the one thing that changes shape.
"""

import linecache

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int8, int16, int32

from test_spmw_tpu_micro import INT8_MAX, TILES, stimulus
from test_spmw_tpu_micro_fixed import NBF

LINK_DEPTH = 2


def lean_engine(
    dim,
    tiles,
    combinational=True,
    link_depth=LINK_DEPTH,
    bias_at_top=False,
    lane_depth=None,
    merged_prologue=False,
    lane_credits=0,
):
    """A `dim x dim` weight-stationary mesh of Gemmini-shaped cells.

    ``bias_at_top`` gives the epilogue Gemmini's shape as well: the bias enters
    as the top of the array's partial sum, as Gemmini's `b` input does, so the
    lane is ReLU, shift and clip with no adder; the shift reads five bits of
    the scale, as `v >> sc(4, 0)` does; and the lane is combinational too.

    ``link_depth=0`` makes every mesh link a bare register (see
    `allo.spmw.abi.fifo_module`): correct here because the array is fed
    without gaps and every cell takes a token from each input every cycle.
    ``lane_depth`` is the link into the lanes, which keeps its handshake.
    ``merged_prologue`` shifts tile 0's weights in on the step loop's first
    `dim` iterations rather than in a loop of their own, so a cell has one
    loop's control instead of two. ``lane_credits`` is the lanes' link credit.
    """
    outs = tiles * dim
    last = dim - 1  # dim is a power of two: `s & last` is the step in a tile

    class PEIO(spmw.Interface):
        """The weight link is an int8: a cell holds one weight, not a file.

        Both ends name the depth. A link takes the deepest either end asked
        for, and one that asked for nothing does not vote.
        """

        a_in = spmw.In(int8, depth=link_depth)
        a_out = spmw.Out(int8, depth=link_depth)
        p_in = spmw.In(int32, depth=link_depth)
        p_out = spmw.Out(int32, depth=link_depth)
        w_in = spmw.In(int8, depth=link_depth)
        w_out = spmw.Out(int8, depth=link_depth)

    class LaneIO(spmw.Interface):
        z_in = spmw.In(int32, depth=link_depth if lane_depth is None else lane_depth)
        y_out = spmw.Out(int32)
        b = spmw.MemIn(int32[NBF])

    mxu = spmw.Topology(
        PEIO,
        grid=(dim, dim),
        link=lambda i, j: {
            PEIO.a_out: spmw.to((i, j + 1), PEIO.a_in),
            PEIO.w_out: spmw.to((i, j + 1), PEIO.w_in),
            PEIO.p_out: spmw.to((i + 1, j), PEIO.p_in),
        },
    )
    lanes = spmw.Grid((dim,))

    @spmw.unit
    def pe(io: PEIO):
        # Tile 0's weights, shifted in before anything computes: Gemmini's
        # preload. After it `nxt` holds this cell's weight.
        nxt: int8 = 0
        for _k in range(dim):
            io.w_out.put(nxt)
            nxt = io.w_in.get()
        cur: int8 = nxt
        for s in range(outs):
            a = io.a_in.get()
            p = io.p_in.get()
            io.a_out.put(a)
            io.p_out.put(p + a * cur)
            # The next tile's weights shift through behind the arithmetic, one
            # a beat, and take over at the tile boundary -- Gemmini's
            # `propagate`, as a step count rather than a flag on the data.
            io.w_out.put(nxt)
            nxt = io.w_in.get()
            if (s & last) == last:
                cur = nxt

    if merged_prologue:

        @spmw.unit
        def pe(io: PEIO):  # pylint: disable=function-redefined
            # One loop: the first `dim` iterations only shift, and the tile
            # boundary at `dim - 1` hands tile 0's weight to `cur` exactly as
            # every later boundary hands over the next tile's.
            nxt: int8 = 0
            cur: int8 = 0
            for s in range(outs + dim):
                if s >= dim:
                    a = io.a_in.get()
                    p = io.p_in.get()
                    io.a_out.put(a)
                    io.p_out.put(p + a * cur)
                io.w_out.put(nxt)
                nxt = io.w_in.get()
                if (s & last) == last:
                    cur = nxt

    if bias_at_top:

        @spmw.unit
        def vpu(io: LaneIO):
            sh: int32 = io.b[1] & 31
            for _m in range(outs):
                acc: int32 = io.z_in.get()
                if acc < 0:  # ReLU
                    acc = 0
                acc = acc >> sh
                if acc > INT8_MAX:
                    acc = INT8_MAX
                io.y_out.put(acc)

        @spmw.fabric
        def engine(
            A: int8[outs, dim],
            W: int8[(tiles + 1) * dim, dim],
            PB: int32[outs, dim],
            Bias: int32[dim, NBF],
            Y: int32[outs, dim],
        ):
            P = spmw.place(pe, on=mxu)
            spmw.pipeline(P, ii=1, combinational=combinational)
            V = spmw.place(vpu, on=lanes)
            spmw.pipeline(V, ii=1, combinational=combinational)
            spmw.shard(Bias, into=V.b)
            spmw.stream_in(A, into=P.a_in, index=(..., P.rows))
            spmw.stream_in(W, into=P.w_in, index=(..., P.rows))
            spmw.stream_in(PB, into=P.p_in, index=(..., P.cols))
            spmw.link(P.p_out, to=V.z_in)
            (lane,) = V.axes
            spmw.gather(Y, from_=V.y_out, index=(..., lane))

    else:

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
                if acc > INT8_MAX:
                    acc = INT8_MAX
                io.y_out.put(acc)

        @spmw.fabric
        def engine(
            A: int8[outs, dim],
            W: int8[(tiles + 1) * dim, dim],
            Bias: int32[dim, NBF],
            Y: int32[outs, dim],
        ):
            P = spmw.place(pe, on=mxu)
            spmw.pipeline(P, ii=1, combinational=combinational)
            V = spmw.place(vpu, on=lanes)
            if lane_credits:
                spmw.pipeline(
                    V,
                    ii=1,
                    registered_links=lane_credits == 1,
                    combinational=lane_credits == 2,
                )
            spmw.shard(Bias, into=V.b)
            spmw.stream_in(A, into=P.a_in, index=(..., P.rows))
            spmw.stream_in(W, into=P.w_in, index=(..., P.rows))
            spmw.stream_in(0, into=P.p_in)
            spmw.link(P.p_out, to=V.z_in)
            (lane,) = V.axes
            spmw.gather(Y, from_=V.y_out, index=(..., lane))

    # Gemmini's mesh spends no DSPs; neither may this one.
    engine.spmw_bind_mul_fabric = True
    return engine


def _generated(name, src, **closure):
    """Define `make` from generated source so `inspect.getsource` can read it.

    A unit is lowered from its source text, which a body with `f * f` weights
    and `3f` ports is not worth writing by hand for every `f`.
    """
    filename = f"<spmw-{name}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    scope = {"int8": int8, "int16": int16, "int32": int32}
    exec(compile(src, filename, "exec"), scope)  # pylint: disable=exec-used
    fn = scope["make"](**closure)
    fn.__name__ = name  # a unit takes its name when it is decorated
    return spmw.unit(fn)


def _tree(terms):
    """A balanced sum, so the adder depth is log2(f) and not f."""
    while len(terms) > 1:
        terms = [
            f"({terms[i]} + {terms[i + 1]})" if i + 1 < len(terms) else terms[i]
            for i in range(0, len(terms), 2)
        ]
    return terms[0]


def _block_source(f):
    """An `f x f` block of lean cells, one loop, straight-line inside it.

    ``n{r}_{c}`` shifts through block row `r` at chain position `c` and
    ``c{r}_{c}`` is the weight in use; column `c`'s partial sum takes the `f`
    products of its column as a balanced tree. For `f = 2` this is::

        io.p0_out.put(p0 + ((a0 * c0_0) + (a1 * c1_0)))
    """
    rows = range(f)
    lines = ["def make(dim, outs, last, IO):", "    def blk(io: IO):"]

    def shift(r, pad):
        out = [f"{pad}io.w{r}_out.put(n{r}_{f - 1})"]
        out += [f"{pad}n{r}_{c} = n{r}_{c - 1}" for c in range(f - 1, 0, -1)]
        return out + [f"{pad}n{r}_0 = io.w{r}_in.get()"]

    lines += [f"        n{r}_{c}: int8 = 0" for r in rows for c in rows]
    lines.append("        for _k in range(dim):")
    for r in rows:
        lines += shift(r, " " * 12)
    lines += [f"        c{r}_{c}: int8 = n{r}_{c}" for r in rows for c in rows]
    lines.append("        for s in range(outs):")
    lines += [f"            a{r} = io.a{r}_in.get()" for r in rows]
    lines += [f"            p{c} = io.p{c}_in.get()" for c in rows]
    lines += [f"            io.a{r}_out.put(a{r})" for r in rows]
    for c in rows:
        dot = _tree([f"a{r} * c{r}_{c}" for r in rows])
        lines.append(f"            io.p{c}_out.put(p{c} + {dot})")
    for r in rows:
        lines += shift(r, " " * 12)
    lines.append("            if (s & last) == last:")
    lines += [f"                c{r}_{c} = n{r}_{c}" for r in rows for c in rows]
    lines.append("    return blk")
    return "\n".join(lines) + "\n"


def _lanes_source(f):
    """Gemmini's epilogue for `f` columns: ReLU, a five-bit shift, the clip."""
    lines = ["def make(outs, INT8_MAX, IO):", "    def lanes(io: IO):"]
    lines.append("        sh: int32 = io.b[1] & 31")
    lines.append("        for _m in range(outs):")
    for c in range(f):
        lines += [
            f"            y{c}: int32 = io.z{c}_in.get()",
            f"            if y{c} < 0:",
            f"                y{c} = 0",
            f"            y{c} = y{c} >> sh",
            f"            if y{c} > INT8_MAX:",
            f"                y{c} = INT8_MAX",
            f"            io.y{c}_out.put(y{c})",
        ]
    lines.append("    return lanes")
    return "\n".join(lines) + "\n"


def fused_engine(dim, tiles, f=2, link_credits=1, link_depth=LINK_DEPTH):
    """`lean_engine` with `f x f` cells to a unit, on a `dim/f` square grid.

    The arithmetic and the weight shift are the lean cell's, `f * f` at a
    time; what fusing changes is how much there is *around* them.  A block has
    `f` links of each kind where its cells had `f * f`, and one loop counter
    and one pipeline's control where they had `f * f`.  `f = dim` is one unit
    with no links inside it at all -- VTA's shape, with the weights in
    registers rather than a scratchpad.

    The epilogue is Gemmini's (see `lean_engine`'s ``bias_at_top``), fused the
    same way: one lane unit per block column.  The operands are the lean
    engine's with the bias at the top, so every `f` is checked against one
    golden result.

    ``link_credits`` is how many link accesses each stage is credited (see
    `allo.spmw.schedule`): one suits a two-stage block, 2x2 and 4x4; 8x8 and
    16x16 are three stages deep and close timing with none.
    """
    if dim % f:
        raise ValueError(f"f={f} does not divide the array side {dim}")
    g = dim // f
    outs = tiles * dim
    rows = range(f)

    ports = {}
    for r in rows:
        ports[f"a{r}_in"] = spmw.In(int8, depth=link_depth)
        ports[f"a{r}_out"] = spmw.Out(int8, depth=link_depth)
        ports[f"w{r}_in"] = spmw.In(int8, depth=link_depth)
        ports[f"w{r}_out"] = spmw.Out(int8, depth=link_depth)
        ports[f"p{r}_in"] = spmw.In(int32, depth=link_depth)
        ports[f"p{r}_out"] = spmw.Out(int32, depth=link_depth)
    BlockIO = spmw.interface(f"Block{f}IO", **ports)
    lane_ports = {"b": spmw.MemIn(int32[NBF])}
    for c in rows:
        lane_ports[f"z{c}_in"] = spmw.In(int32, depth=link_depth)
        lane_ports[f"y{c}_out"] = spmw.Out(int32)
    LanesIO = spmw.interface(f"Lanes{f}IO", **lane_ports)

    def link(i, j):
        port = lambda name: getattr(BlockIO, name)  # noqa: E731
        out = {}
        for r in rows:
            out[port(f"a{r}_out")] = spmw.to((i, j + 1), port(f"a{r}_in"))
            out[port(f"w{r}_out")] = spmw.to((i, j + 1), port(f"w{r}_in"))
            out[port(f"p{r}_out")] = spmw.to((i + 1, j), port(f"p{r}_in"))
        return out

    blocks = spmw.Topology(BlockIO, grid=(g, g), link=link)
    lanes = spmw.Grid((g,))
    blk = _generated(
        f"blk{f}", _block_source(f), dim=dim, outs=outs, last=dim - 1, IO=BlockIO
    )
    lane = _generated(
        f"lanes{f}", _lanes_source(f), outs=outs, INT8_MAX=INT8_MAX, IO=LanesIO
    )

    @spmw.fabric
    def engine(
        A: int8[outs, dim],
        W: int8[(tiles + 1) * dim, dim],
        PB: int32[outs, dim],
        Bias: int32[g, NBF],
        Y: int32[outs, dim],
    ):
        P = spmw.place(blk, on=blocks)
        spmw.pipeline(
            P, ii=1, registered_links=link_credits == 1, combinational=link_credits == 2
        )
        V = spmw.place(lane, on=lanes)
        spmw.pipeline(
            V, ii=1, registered_links=link_credits == 1, combinational=link_credits == 2
        )
        spmw.shard(Bias, into=V.b)
        (col,) = V.axes
        for r in rows:
            spmw.stream_in(A, into=getattr(P, f"a{r}_in"), index=(..., f * P.rows + r))
            spmw.stream_in(W, into=getattr(P, f"w{r}_in"), index=(..., f * P.rows + r))
            spmw.stream_in(PB, into=getattr(P, f"p{r}_in"), index=(..., f * P.cols + r))
            spmw.link(getattr(P, f"p{r}_out"), to=getattr(V, f"z{r}_in"))
            spmw.gather(Y, from_=getattr(V, f"y{r}_out"), index=(..., f * col + r))

    engine.spmw_bind_mul_fabric = True
    return engine


def micro_fused_of(
    size, f=2, tiles=TILES, seed=0, link_credits=1, link_depth=LINK_DEPTH
):
    """The fused engine with the shared stimulus attached."""
    A, B, bias, shift, want = stimulus(size, tiles, seed)
    a, w, top, consts = lean_operands(size, A, B, bias, shift, bias_at_top=True)
    engine = fused_engine(
        size, tiles, f=f, link_credits=link_credits, link_depth=link_depth
    )
    engine.spmw_operands = {
        "A": a,
        "W": w,
        "PB": top,
        "Bias": np.ascontiguousarray(consts[: size // f]),
    }
    engine.spmw_tokens_per_transform = size * size
    engine.spmw_micro = {
        "size": size,
        "tiles": tiles,
        "shift": int(shift),
        "fused": f,
        "link_credits": link_credits,
        "link_depth": link_depth,
        "expected": want.reshape(tiles * size, size).astype(np.int32),
    }
    return engine


def shift_stream(B):
    """Each row's weights, a tile at a time, in the order the shift needs.

    Row `k` sends tile `t`'s weights for columns ``S-1 .. 0``: the first token
    in travels furthest.  One tile of zeros closes the stream, because the last
    tile's beats shift too and there is no next tile to shift in.
    """
    tiles, size = B.shape[0], B.shape[1]
    W = np.zeros(((tiles + 1) * size, size), dtype=np.int8)
    for t in range(tiles):
        W[t * size : (t + 1) * size, :] = B[t][:, ::-1].T
    return W


def lean_operands(size, A, B, bias, shift, bias_at_top=False):
    consts = np.zeros((size, NBF), dtype=np.int32)
    consts[:, 0] = bias
    consts[:, 1] = shift
    a = A.reshape(-1, size)
    if not bias_at_top:
        return a, shift_stream(B), consts
    # The bias as every row's top-of-array partial sum, Gemmini's `b`.
    top = np.ascontiguousarray(np.broadcast_to(bias.astype(np.int32), a.shape))
    return a, shift_stream(B), top, consts


def micro_lean_of(
    size,
    tiles=TILES,
    seed=0,
    combinational=True,
    link_depth=LINK_DEPTH,
    bias_at_top=False,
    **options,
):
    """The lean engine with the shared stimulus attached.

    ``options`` go to `lean_engine`: ``lane_depth``, ``merged_prologue``,
    ``lane_credits``.
    """
    A, B, bias, shift, want = stimulus(size, tiles, seed)
    engine = lean_engine(
        size,
        tiles,
        combinational=combinational,
        link_depth=link_depth,
        bias_at_top=bias_at_top,
        **options,
    )
    names = ("A", "W", "PB", "Bias") if bias_at_top else ("A", "W", "Bias")
    engine.spmw_operands = dict(
        zip(names, lean_operands(size, A, B, bias, shift, bias_at_top))
    )
    engine.spmw_tokens_per_transform = size * size
    engine.spmw_micro = {
        "size": size,
        "tiles": tiles,
        "shift": int(shift),
        "lean": True,
        "combinational": combinational,
        "link_depth": link_depth,
        "bias_at_top": bias_at_top,
        **options,
        "expected": want.reshape(tiles * size, size).astype(np.int32),
    }
    return engine


# -- tests --------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_four_tiles_bit_exact(target):
    size, tiles = 4, 4
    A, B, bias, shift, want = stimulus(size, tiles)
    Y = np.zeros((tiles * size, size), dtype=np.int32)
    spmw.build(lean_engine(size, tiles), target=target)(
        *lean_operands(size, A, B, bias, shift), Y
    )
    np.testing.assert_array_equal(Y, want.reshape(-1, size).astype(np.int32))


@pytest.mark.parametrize("size", [4, 8, 16])
def test_the_measured_configuration_matches_the_golden(size):
    """The cosim checks RTL against `ref`; this checks `ref` against the golden."""
    engine = micro_lean_of(size)
    want = engine.spmw_micro["expected"]
    Y = np.zeros(want.shape, dtype=np.int32)
    spmw.build(engine, target="ref")(
        *[engine.spmw_operands[n] for n in ("A", "W", "Bias")], Y
    )
    np.testing.assert_array_equal(Y, want)


def test_it_is_the_fixed_engines_workload():
    """Same activations, bias and shift; the same weights in a new order."""
    from test_spmw_tpu_micro_fixed import micro_fixed_of

    lean, fixed = micro_lean_of(8), micro_fixed_of(8)
    np.testing.assert_array_equal(
        lean.spmw_micro["expected"], fixed.spmw_micro["expected"]
    )
    np.testing.assert_array_equal(lean.spmw_operands["A"], fixed.spmw_operands["A"])
    np.testing.assert_array_equal(
        lean.spmw_operands["Bias"], fixed.spmw_operands["Bias"]
    )


def test_the_schedule_is_a_directive_not_a_different_design():
    """`combinational` changes how the cell is scheduled, never what it means."""
    import allo.spmw.schedule as sched
    from allo.spmw.lower_mlir import render_module

    on = spmw.elaborate(lean_engine(4, 4, combinational=True))
    off = spmw.elaborate(lean_engine(4, 4, combinational=False))
    assert sched.link_credits(on.placements[0]) == 2
    assert sched.link_credits(off.placements[0]) == 0
    assert sched.link_credits(on.placements[1]) == 0, "the lanes keep their stages"
    fused = spmw.elaborate(fused_engine(4, 4, f=2))
    assert [sched.link_credits(p) for p in fused.placements] == [1, 1]
    assert render_module(on) == render_module(off)


@pytest.mark.parametrize("f", [2, 4])
@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_fused_blocks_bit_exact(target, f):
    """`f * f` cells to a unit compute what `f * f` units did."""
    size, tiles = 4, 4
    A, B, bias, shift, want = stimulus(size, tiles)
    a, w, top, consts = lean_operands(size, A, B, bias, shift, bias_at_top=True)
    Y = np.zeros((tiles * size, size), dtype=np.int32)
    spmw.build(fused_engine(size, tiles, f=f), target=target)(
        a, w, top, np.ascontiguousarray(consts[: size // f]), Y
    )
    np.testing.assert_array_equal(Y, want.reshape(-1, size).astype(np.int32))


@pytest.mark.parametrize("size,f", [(8, 2), (16, 2), (16, 4), (16, 8), (16, 16)])
def test_the_fused_configuration_matches_the_golden(size, f):
    engine = micro_fused_of(size, f=f)
    want = engine.spmw_micro["expected"]
    Y = np.zeros(want.shape, dtype=np.int32)
    ops = [engine.spmw_operands[n] for n in ("A", "W", "PB", "Bias")]
    spmw.build(engine, target="ref")(*ops, Y)
    np.testing.assert_array_equal(Y, want)


def test_the_generated_block_is_the_lean_cell_at_f_1():
    """The generator at `f = 1` writes the hand-written cell, name for name."""
    src = _block_source(1)
    assert "io.p0_out.put(p0 + a0 * c0_0)" in src
    assert "n0_0 = io.w0_in.get()" in src and "io.w0_out.put(n0_0)" in src


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_the_bias_can_enter_at_the_top(target):
    """Gemmini's epilogue shape computes the same bytes."""
    size, tiles = 4, 4
    A, B, bias, shift, want = stimulus(size, tiles)
    Y = np.zeros((tiles * size, size), dtype=np.int32)
    spmw.build(lean_engine(size, tiles, bias_at_top=True), target=target)(
        *lean_operands(size, A, B, bias, shift, bias_at_top=True), Y
    )
    np.testing.assert_array_equal(Y, want.reshape(-1, size).astype(np.int32))


def test_the_bias_at_the_top_matches_the_golden_at_16():
    engine = micro_lean_of(16, bias_at_top=True)
    want = engine.spmw_micro["expected"]
    Y = np.zeros(want.shape, dtype=np.int32)
    spmw.build(engine, target="ref")(
        *[engine.spmw_operands[n] for n in ("A", "W", "PB", "Bias")], Y
    )
    np.testing.assert_array_equal(Y, want)


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_the_merged_prologue_computes_the_same(target):
    """Shifting tile 0 in on the step loop changes the control, not the sum."""
    size, tiles = 4, 4
    A, B, bias, shift, want = stimulus(size, tiles)
    Y = np.zeros((tiles * size, size), dtype=np.int32)
    engine = lean_engine(size, tiles, merged_prologue=True, lane_credits=1)
    spmw.build(engine, target=target)(*lean_operands(size, A, B, bias, shift), Y)
    np.testing.assert_array_equal(Y, want.reshape(-1, size).astype(np.int32))


def test_bare_links_are_the_mesh_links_only():
    """Depth 0 inside the array; the lanes' link keeps its handshake."""
    import re

    from allo.spmw import rtl

    engine = lean_engine(4, 4, link_depth=0, lane_depth=2)
    sv = rtl.StructuralEmitter(spmw.elaborate(engine)).fabric()
    got = {
        fam: d
        for d, fam in re.findall(r"\.DEPTH\((\d+)\)\) u \(.*?\.din\(([a-z_]+?)_din", sv)
    }
    assert {got[f] for f in got if f.startswith("pe_")} == {"0"}, got
    assert {got[f] for f in got if f.startswith("vpu_")} == {"2"}, got
