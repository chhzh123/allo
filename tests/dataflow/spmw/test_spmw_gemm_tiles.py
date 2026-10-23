# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One output-stationary array, a GEMM many times its size.

`test_spmw_autosa_match.py` computes one ``S x S x S`` tile on ``S x S`` cells.
This is the same array given ``C = A B`` with ``A`` of ``M x K`` and ``B`` of
``K x N``: ``(M / S) * (N / S)`` tiles, each ``K`` steps long, one after another
with no cycle between them.

Three things keep a cell from ever waiting on a tile boundary.

* **The stream says where a tile ends.** An operand of ``A`` travels as its
  int8 value and two marks, as a bus carries TLAST: `LAST` on a tile's final
  step and `END` on the launch's. A cell counts no steps; on `LAST` it hands
  its sum on and starts the next tile's from zero in the same step.
* **Results leave during the next tile.** A column's sums go down the column,
  as in the matched design, but a cell does not stop to pass them: in each of
  the steps after a `LAST` it forwards one sum from above beside that step's
  multiply-add. This needs a tile at least as long as the array is tall,
  ``K >= S``.
* **A cell is one state.** A row runs one cycle behind the row above, so a sum
  should go down a row in one cycle too: the step's multiply-add, the sum it
  hands on and the one it takes are all in one state. The launch's last sums
  go down in the same loop, once `END` has passed: a second loop for them
  would be a second machine in every cell.

Measured in RTL on 16 tiles (`one_state` and `sum_depth` take the last two
away): the array as written finishes in its steps plus ``3S + 7`` cycles, 275
at 4x4 and 543 at 8x8. Left to pipeline, the cell is five stages deep, four
cycles later to its first sum, and done at 282 and 554. With the links that
carry the sums left two deep it is done at 290 and 618 -- 0.9 and 4.7 cycles
lost a tile.

The marks are not in memory. A row's operands arrive as plain int8 behind a
five-byte header -- ``K - 1`` and the tile count less one -- and one small unit
a row, `frame`, counts and marks them. Those are the only counters that depend
on the workload, and they are loaded, not compiled: one build of the array
runs every ``M``, ``K`` and ``N``, which `test_one_array_for_every_workload`
checks.

The host lays the operands out in the order the array reads them (`streams`):
row tile after row tile, each against every column tile, a step a token. That
is the layout every SPMW kernel takes, and it is ``N / S`` copies of ``A`` and
``M / S`` of ``B``.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int8, int16, int32, uint1, uint8, uint16

#: A framed operand's two marks, above its int8 value.
LAST, END = 1 << 8, 1 << 9
#: The tokens a row's stream opens with.
HEADER = 5


def gemm_tiles_of(size, na, nb, nc, one_state=True, sum_depth=4):
    """The ``size x size`` array, with edge tensors of the given lengths.

    The lengths size the harness's streams, not the hardware. ``one_state`` and
    ``sum_depth`` are two of the things the array has so as to lose nothing at
    a tile boundary; each can be taken away to measure what it is worth.
    """
    n = size

    class MacIO(spmw.Interface):
        west = spmw.In(int16)  # an operand of A, framed
        north = spmw.In(int8)  # an operand of B
        east = spmw.Out(int16)
        south = spmw.Out(int8)
        # The sums of the rows above. The row above is two steps into handing
        # them down when this cell takes its first, and a two-entry slice says
        # it is full a cycle after it is: a third place keeps the column moving.
        c_in = spmw.In(int32, depth=sum_depth)
        c_out = spmw.Out(int32)

    class FrameIO(spmw.Interface):
        raw = spmw.In(int8)  # the header, then this row's operands
        out = spmw.Out(int16)

    mesh = spmw.Topology(
        MacIO,
        grid=(n, n),
        link=lambda i, j: {
            MacIO.east: spmw.to((i, j + 1), MacIO.west),
            MacIO.south: spmw.to((i + 1, j), MacIO.north),
            MacIO.c_out: spmw.to((i + 1, j), MacIO.c_in),
        },
    )
    edge = spmw.Topology(FrameIO, grid=(n,), link=lambda i: {})

    @spmw.unit
    def pe(io: MacIO, site: spmw.Site):
        row, _col = site.rank
        above: uint8 = row  # sized: a bare `row` counts in 32 bits
        acc: int32 = 0
        rem: uint8 = 0
        tail: uint1 = 0
        go: uint1 = 1
        while go != 0:
            if tail != 0:  # the launch is over: what is still coming down
                io.c_out.put(io.c_in.get())
                if rem == 1:
                    go = 0
                rem = rem - 1
            else:
                t: int16 = io.west.get()
                b: int8 = io.north.get()
                io.east.put(t)
                io.south.put(b)
                a: int8 = ((t & 255) ^ 128) - 128
                s: int32 = acc + a * b
                if ((t >> 8) & 1) != 0:  # LAST: this tile's sum is whole
                    io.c_out.put(s)
                    acc = 0
                    rem = above
                else:
                    acc = s
                    if rem != 0:  # a sum of the tile before, from above
                        io.c_out.put(io.c_in.get())
                        rem = rem - 1
                if ((t >> 9) & 1) != 0:  # END
                    tail = 1
                    if above == 0:
                        go = 0

    @spmw.unit
    def frame(io: FrameIO):
        # `kc` counts a tile's steps down and `tc` the tiles. `klast` and
        # `tlast` say a counter is on its last and are kept as flags, set as
        # the counters move, so no token waits on a comparison.
        h0: int8 = io.raw.get()
        h1: int8 = io.raw.get()
        h2: int8 = io.raw.get()
        h3: int8 = io.raw.get()
        h4: int8 = io.raw.get()
        kn: uint16 = (h0 & 255) | ((h1 & 255) << 8)
        tc: int32 = (h2 & 255) | ((h3 & 255) << 8) | ((h4 & 255) << 16)
        kc: uint16 = kn
        kone: uint1 = 0
        if kn == 0:
            kone = 1
        klast: uint1 = kone
        tlast: uint1 = 0
        if tc == 0:
            tlast = 1
        go: uint1 = 1
        while go != 0:
            v: int8 = io.raw.get()
            u: int16 = v & 255
            if klast != 0:
                u = u | 256
                kc = kn
                klast = kone
                if tlast != 0:
                    u = u | 512
                    go = 0
                else:
                    if tc == 1:
                        tlast = 1
                    tc = tc - 1
            else:
                if kc == 1:
                    klast = 1
                kc = kc - 1
            io.out.put(u)

    @spmw.fabric
    def g(At: int8[na, n], Bt: int8[nb, n], Ct: int32[nc, n]):
        P = spmw.place(pe, on=mesh)
        if one_state:
            spmw.pipeline(P, ii=1, combinational=True)
        F = spmw.place(frame, on=edge)
        (lane,) = F.axes
        spmw.stream_in(At, into=F.raw, index=(..., lane))
        spmw.link(F.out, to=P.west)
        spmw.stream_in(Bt, into=P.north, index=(..., P.cols))
        # Row 0 has no row above and never reads this, but the port is in its
        # body and has to be covered.
        spmw.stream_in(0, into=P.c_in)
        # A tile's column arrives bottom row first.
        spmw.gather(Ct, from_=P.c_out, index=(..., P.cols))

    return g


# -- the host's side ----------------------------------------------------------


def streams(a, b, size):
    """``A`` and ``B`` as the array reads them: a step a token, tile after tile.

    Tile ``(mt, nt)`` is rows ``mt * size ..`` of ``A`` against columns
    ``nt * size ..`` of ``B``; the tiles run ``mt`` outermost.
    """
    (m, k), (_k, n) = a.shape, b.shape
    if m % size or n % size:
        raise ValueError(f"{m} x {n} results do not tile a {size} x {size} array")
    if k < size:
        raise ValueError(f"a tile of {k} steps is shorter than the array is tall")
    tm, tn = m // size, n // size
    tiles = tm * tn
    a_blocks = a.reshape(tm, size, k).transpose(0, 2, 1)  # [mt][step][row]
    b_blocks = b.reshape(k, tn, size).transpose(1, 0, 2)  # [nt][step][col]
    at = np.repeat(a_blocks, tn, axis=0).reshape(tiles * k, size)
    bt = np.tile(b_blocks, (tm, 1, 1)).reshape(tiles * k, size)
    header = np.array(
        [k - 1, (k - 1) >> 8, tiles - 1, (tiles - 1) >> 8, (tiles - 1) >> 16]
    )
    head = np.repeat((header & 255).astype(np.uint8)[:, None], size, axis=1)
    at = np.concatenate([head.view(np.int8), at.astype(np.int8)])
    return np.ascontiguousarray(at), np.ascontiguousarray(bt.astype(np.int8))


def drained(c, size):
    """``C`` as the array hands it back: a tile's column, bottom row first."""
    m, n = c.shape
    tm, tn = m // size, n // size
    blocks = c.reshape(tm, size, tn, size).transpose(0, 2, 1, 3)  # [mt][nt][row][col]
    return np.ascontiguousarray(blocks[:, :, ::-1, :].reshape(tm * tn * size, size))


def gathered(ct, m, n, size):
    """The inverse of `drained`."""
    tm, tn = m // size, n // size
    blocks = ct.reshape(tm, tn, size, size)[:, :, ::-1, :]
    return np.ascontiguousarray(blocks.transpose(0, 2, 1, 3).reshape(m, n))


def tiles_workload(size, m, k=None, n=None, seed=0, **form):
    """``C = A B`` on the ``size x size`` array, as one launch with its golden.

    The operands span int8 and hold both extremes. ``form`` is passed to
    `gemm_tiles_of`.
    """
    k = m if k is None else k
    n = m if n is None else n
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, (m, k)).astype(np.int8)
    b = rng.integers(-128, 128, (k, n)).astype(np.int8)
    a[0, 0], a[-1, -1], b[0, 0], b[-1, -1] = -128, 127, 127, -128
    at, bt = streams(a, b, size)
    want = drained(a.astype(np.int32) @ b.astype(np.int32), size)
    g = gemm_tiles_of(size, len(at), len(bt), len(want), **form)
    g.spmw_operands = {"At": at, "Bt": bt}
    g.spmw_expected = {"Ct": want}
    # Every step, the fill and drain of the array, and room to be wrong in.
    g.spmw_cosim_cycles = len(bt) + 64 * size + 10000
    g.spmw_tiles = {"a": a, "b": b, "steps": len(bt), "ideal": m * k * n // size**2}
    return g


# -- tests --------------------------------------------------------------------


def _run(g, target):
    want = g.spmw_expected["Ct"]
    ct = np.zeros(want.shape, dtype=np.int32)
    spmw.build(g, target=target)(g.spmw_operands["At"], g.spmw_operands["Bt"], ct)
    np.testing.assert_array_equal(ct, want)
    a, b = g.spmw_tiles["a"], g.spmw_tiles["b"]
    size = want.shape[1]
    np.testing.assert_array_equal(
        gathered(ct, a.shape[0], b.shape[1], size),
        a.astype(np.int32) @ b.astype(np.int32),
    )


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_one_tile_matches_numpy(target):
    _run(tiles_workload(4, 4), target)


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_four_tiles_match_numpy(target):
    _run(tiles_workload(4, 8), target)


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_a_rectangular_gemm_matches_numpy(target):
    """Six tiles of eleven steps: nothing has to be a multiple of anything."""
    _run(tiles_workload(2, 4, k=11, n=6), target)


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_a_tile_as_short_as_the_array_is_tall(target):
    """``K = S``: the column's sums clear with no step to spare."""
    _run(tiles_workload(4, 8, k=4), target)


@pytest.mark.parametrize("form", [{"one_state": False}, {"sum_depth": 2}])
def test_the_slower_forms_compute_the_same(form):
    """A pipelined cell, or two-deep sum links: slower in hardware, not wrong."""
    _run(tiles_workload(4, 8, **form), "ref")


def test_a_tile_shorter_than_the_array_is_refused():
    with pytest.raises(ValueError, match="shorter than the array is tall"):
        tiles_workload(4, 8, k=3)


def test_the_layout_repeats_the_operands_and_marks_nothing():
    """The image is plain int8: ``N / S`` copies of ``A``, ``M / S`` of ``B``."""
    g = tiles_workload(2, 4, k=3, n=6)
    a, b = g.spmw_tiles["a"], g.spmw_tiles["b"]
    at, bt = g.spmw_operands["At"], g.spmw_operands["Bt"]
    assert at.dtype == np.int8 and bt.dtype == np.int8
    assert at.shape == (HEADER + 6 * 3, 2) and bt.shape == (6 * 3, 2)
    assert list(at[:HEADER, 0]) == [2, 0, 5, 0, 0]  # K - 1, then tiles - 1
    # tile 4 is row tile 1 against column tile 1
    np.testing.assert_array_equal(at[HEADER + 4 * 3 :][:3], a[2:4, :].T)
    np.testing.assert_array_equal(bt[4 * 3 :][:3], b[:, 2:4])


def test_the_frames_are_called_before_the_cells():
    """A region's processes are emitted producer first.

    Their order means nothing to the hardware or to SPMW's own simulator,
    which runs them at once. Vitis HLS's C simulation runs them one after
    another in the order they are called, and a cell called before the
    `frame` that feeds it reads an empty stream.
    """
    # pylint: disable=import-outside-toplevel
    from allo.spmw.lower_mlir import Program

    names = [plan.name for plan in Program(spmw.elaborate(tiles_workload(4, 8))).maps]
    assert names.index("frame") < names.index("pe"), names


def test_one_array_for_every_workload():
    """Two GEMMs of different shapes compile to the same roles and fabric."""
    from test_spmw_ptpu import hardware  # pylint: disable=import-outside-toplevel

    one = hardware(tiles_workload(4, 8))
    other = hardware(tiles_workload(4, 12, k=20, n=8))
    assert sorted(one) == sorted(other)
    for role, code in one.items():
        assert code == other[role], f"{role} differs between the two workloads"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
