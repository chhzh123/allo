# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The systolic GEMM built to match what AutoSA generates, structurally.

`test_spmw_daisy.py` chains the *drain*, which was the largest mismatch. This
closes the other two: operands arrive through **daisy-chained distribution
networks** rather than one edge stream per row and column, and the arithmetic is
int8 into an int32 accumulator, as AutoSA's is.

The result has the same three-part shape as `kernel_kernel.cpp` in
`examples/spmw/autosa/int8`:

| AutoSA | here |
|---|---|
| `A_IO_L2_in` chain | the `Fa` placement |
| `B_IO_L2_in` chain | the `Fb` placement |
| `PE_wrapper` mesh | the `P` placement |
| `C_drain_IO_L1_out` chain | `c_in`/`c_out` on the mesh |

What is still missing is the DRAM interface — AutoSA's `A_IO_L3` and the AXI
masters. This fabric ends at its edge streams, which is the gap
`SPMW_EXPERIMENTS.md` E14 is about.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int8, int32

SIZE = 4


def autosa_match_of(size):
    """The matched design at a chosen size."""
    n = size

    class MacIO(spmw.Interface):
        west = spmw.In(int8)
        north = spmw.In(int8)
        east = spmw.Out(int8)
        south = spmw.Out(int8)
        # A *scalar* drain, as AutoSA's is (`typedef int C_t1`,
        # `hls::stream<int>`): each PE emits its own result and then forwards
        # what came from above, one at a time. Packing the whole column into one
        # wide token instead -- which is what Allo's daisy-chain design does --
        # puts a 512-bit register in every PE and costs ~5x the area.
        c_in = spmw.In(int32)
        c_out = spmw.Out(int32)

    class FeedIO(spmw.Interface):
        """One link of a distribution chain: take a packed vector, keep one
        lane, pass the rest along."""

        up = spmw.In(int8[n])
        down = spmw.Out(int8[n])
        lane = spmw.Out(int8)

    mesh = spmw.Topology(
        MacIO,
        grid=(n, n),
        link=lambda i, j: {
            MacIO.east: spmw.to((i, j + 1), MacIO.west),
            MacIO.south: spmw.to((i + 1, j), MacIO.north),
            MacIO.c_out: spmw.to((i + 1, j), MacIO.c_in),
        },
    )
    chain = spmw.Topology(
        FeedIO,
        grid=(n,),
        link=lambda i: {FeedIO.down: spmw.to((i + 1,), FeedIO.up)},
    )

    @spmw.unit
    def pe(io: MacIO, site: spmw.Site):
        row, _col = site.rank
        acc: int32 = 0
        for k in range(n):
            a = io.west.get()
            b = io.north.get()
            acc += a * b
            io.east.put(a)
            io.south.put(b)
        io.c_out.put(acc)
        for _i in range(row):
            io.c_out.put(io.c_in.get())

    @spmw.unit
    def feed(io: FeedIO, site: spmw.Site):
        (slot,) = site.rank
        for k in range(n):
            packed: int8[n] = io.up.get()
            io.lane.put(packed[slot])
            io.down.put(packed)

    @spmw.fabric
    def g(At: int8[n, n], Bt: int8[n, n], Ct: int32[n, n]):
        P = spmw.place(pe, on=mesh)
        Fa = spmw.place(feed, on=chain)
        Fb = spmw.place(feed, on=chain)
        # One packed vector per step enters the head of each chain; every link
        # keeps its own lane and forwards the rest.
        spmw.stream_in(At, into=Fa.up, index=(...,))
        spmw.stream_in(Bt, into=Fb.up, index=(...,))
        spmw.link(Fa.lane, to=P.west)
        spmw.link(Fb.lane, to=P.north)
        spmw.stream_in(0, into=P.c_in)
        # Each bottom PE emits n results: its own first, then what it forwards
        # from above -- so a column arrives in reverse row order.
        spmw.gather(Ct, from_=P.c_out, index=(P.cols, ...))

    g.spmw_parts = (MacIO, FeedIO, mesh, chain, pe, feed)
    return g


autosa_match = autosa_match_of(SIZE)


def _operands(size=SIZE, seed=9):
    """`At[k]` is column k of A; `Bt[k]` is row k of B — one token per step."""
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 4, size=(size, size)).astype(np.int8)
    b = rng.integers(-4, 4, size=(size, size)).astype(np.int8)
    return a, b, np.ascontiguousarray(a.T), b, np.zeros((size, size), dtype=np.int32)


def _unpack(ct):
    """`ct[c][t]` is column c's t-th arrival, and they arrive bottom row first."""
    return ct[:, ::-1].T


def test_reference_matches_numpy():
    a, b, at, bt, ct = _operands()
    spmw.build(autosa_match, target="ref")(at, bt, ct)
    np.testing.assert_array_equal(_unpack(ct), a.astype(np.int32) @ b.astype(np.int32))


def test_simulator_matches_numpy():
    a, b, at, bt, ct = _operands()
    spmw.build(autosa_match, target="simulator")(at, bt, ct)
    np.testing.assert_array_equal(_unpack(ct), a.astype(np.int32) @ b.astype(np.int32))


def test_the_edge_is_two_chains_and_a_drain():
    """Three placements, and an array edge that does not grow with the mesh.

    A plain mesh needs an edge stream per row, per column, and per site. This
    needs the head of two chains and the foot of one, which is what lets AutoSA
    put a DRAM interface behind a fixed number of ports.
    """
    from allo.spmw import rtl

    emitter = rtl.StructuralEmitter(spmw.elaborate(autosa_match))
    assert len(emitter.placements()) == 3, "mesh + two feed chains"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
