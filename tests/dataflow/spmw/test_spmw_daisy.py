# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The daisy-chained systolic GEMM — `tests/dataflow/test_daisy_chain_gemm.py`.

The plain mesh in `test_spmw_gemm.py` gives every site its own `c` port, so a
16x16 array drains through 256 separate edge streams.  Real systolic arrays do
not do that, and neither does AutoSA: results are *chained* out, one packed
column at a time, so the array's edge is O(N) rather than O(N^2).

This is that design.  Each PE accumulates as before, then takes the partial
column arriving from the north, drops its own result into its own slot, and
passes it south.  The bottom row's columns are what leaves the array.  It is
AutoSA's `C_drain_IO_L1_out` network expressed as a link rather than as
generated glue, and it is what makes the two comparable.

Writing one's own slot means the body reads its coordinate, so this is also the
first design here that needs the unit to take its position as an input — see
`allo.spmw.rtl.CoordPort`.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int16

M, N, K = 4, 4, 4


def daisy_of(size):
    """The chained-drain mesh at a chosen size; the grid is a parameter."""
    n = size

    class MacIO(spmw.Interface):
        west = spmw.In(int16)
        north = spmw.In(int16)
        east = spmw.Out(int16)
        south = spmw.Out(int16)
        # The drain chain: a whole column of results per token, not one scalar.
        c_in = spmw.In(int16[n])
        c_out = spmw.Out(int16[n])

    # `spmw.mesh` wires the NSEW four by convention and knows nothing about the
    # drain pair, so the topology is spelled out: the same four links plus the
    # chain carrying a partial column from each row to the one below it.
    topo = spmw.Topology(
        MacIO,
        grid=(n, n),
        link=lambda i, j: {
            MacIO.east: spmw.to((i, j + 1), MacIO.west),
            MacIO.south: spmw.to((i + 1, j), MacIO.north),
            MacIO.c_out: spmw.to((i + 1, j), MacIO.c_in),
        },
    )

    @spmw.unit
    def pe(io: MacIO, site: spmw.Site):
        row, _col = site.rank
        acc: int16 = 0
        for k in range(n):
            a = io.west.get()
            b = io.north.get()
            acc += a * b
            io.east.put(a)
            io.south.put(b)
        column: int16[n] = io.c_in.get()
        column[row] = acc
        io.c_out.put(column)

    @spmw.fabric
    def g(A: int16[n, n], B: int16[n, n], Ct: int16[n, n]):
        P = spmw.place(pe, on=topo)
        spmw.stream_in(A, into=P.west, index=(P.rows, ...))
        spmw.stream_in(B, into=P.north, index=(..., P.cols))
        # The chain starts empty at the top row and leaves at the bottom. A seed
        # rather than a tensor: there is nothing to read, and AutoSA's drain
        # starts empty too.
        spmw.stream_in(0, into=P.c_in)
        spmw.gather(Ct, from_=P.c_out)

    g.spmw_parts = (MacIO, topo, pe)
    return g


daisy_gemm = daisy_of(M)
MacIO, topo, pe = daisy_gemm.spmw_parts


def _operands(seed=5):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 8, size=(M, K)).astype(np.int16)
    b = rng.integers(0, 8, size=(K, N)).astype(np.int16)
    # The chain hands out one column of C per grid column, so the result arrives
    # transposed -- column j of C is row j here.
    return a, b, np.zeros((N, M), dtype=np.int16)


def test_reference_matches_numpy():
    a, b, ct = _operands()
    spmw.build(daisy_gemm, target="ref")(a, b, ct)
    want = (a.astype(np.int32) @ b.astype(np.int32)).astype(np.int16)
    np.testing.assert_array_equal(ct.T, want)


def test_simulator_matches_numpy():
    a, b, ct = _operands()
    spmw.build(daisy_gemm, target="simulator")(a, b, ct)
    want = (a.astype(np.int32) @ b.astype(np.int32)).astype(np.int16)
    np.testing.assert_array_equal(ct.T, want)


def test_a_stream_drain_off_an_edge_needs_only_the_edge():
    """The bare identity is over the *bundle*, not the grid.

    `c_out` is consumed by the chain everywhere but the bottom row, so its
    bundle is that row alone and the tensor it drains into is shaped like the
    row plus the block. Using the placement grid here demanded a tensor shaped
    like the whole mesh, which is what this design first hit.
    """
    P = spmw.place(pe, on=topo)
    assert P.c_out.shape == (N,), P.c_out.shape
    assert sorted(P.c_out.sites) == [(M - 1, j) for j in range(N)]
    assert P.c_in.shape == (N,)
    assert sorted(P.c_in.sites) == [(0, j) for j in range(N)]


def test_the_drain_is_a_chain_not_a_port_per_site():
    """O(N) edge streams instead of O(N^2) — the point of the design.

    The plain mesh exports one `c` channel per site; this exports one column per
    grid column. That is the difference between AutoSA's drain network and a
    fanout our fabric would otherwise have to carry.
    """
    from allo.spmw import rtl

    chained = rtl.cost(spmw.elaborate(daisy_gemm))
    from test_spmw_rolled import gemm_of

    plain = rtl.cost(spmw.elaborate(gemm_of(M)))
    assert chained["instances"] == plain["instances"] == M * N
    # the plain mesh drains every site individually
    assert plain["streams"] > chained["streams"], (plain, chained)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
