# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The systolic GEMM the other way round: the weights stay, the results move.

`test_spmw_gemm_int8.py` holds a *result* in every cell and streams both
operands past it.  This holds a *weight* in every cell -- cell (k, n) keeps
B[k][n] -- sends the activations east along the rows and the partial sums south
down the columns, so a row of C leaves the bottom of the array every step.  The
arithmetic is the same (int8 into int32), and so is the array: S x S cells, one
multiplier each.

The weights get there the way they would in hardware: down their own column,
once, before the first activation.  A cell keeps the first weight it sees and
hands the rest on.

Nothing has to be laid out for it: a row of A is a step of activations, a row of
B is a step of weights, and a step of results is a row of C.  So the array's
edge is three bindings, and packaged as a kernel it is three memory movers.  A
weight-stationary array's wavefront is a diagonal -- row k wants its activations
k cells later than row 0, and column c's results leave c cells after column
0's -- and the edge streams, which hold a launch, are where that skew waits.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int8, int32, uint8

SIZE = 4


def gemm_ws_of(size):
    """The weight-stationary mesh at a chosen size, a stream at every edge."""
    n = size
    assert n <= 256, "a cell counts the rows below it in eight bits"

    class WsIO(spmw.Interface):
        a_in = spmw.In(int8)
        a_out = spmw.Out(int8)
        w_in = spmw.In(int8)
        w_out = spmw.Out(int8)
        p_in = spmw.In(int32)
        p_out = spmw.Out(int32)

    mesh = spmw.Topology(
        WsIO,
        grid=(n, n),
        link=lambda k, c: {
            WsIO.a_out: spmw.to((k, c + 1), WsIO.a_in),
            WsIO.w_out: spmw.to((k + 1, c), WsIO.w_in),
            WsIO.p_out: spmw.to((k + 1, c), WsIO.p_in),
        },
    )

    @spmw.unit
    def mac(io: WsIO, site: spmw.Site):
        row, _col = site.rank
        below: uint8 = n - 1 - row  # sized: a bare `row` counts in 32 bits
        w: int8 = io.w_in.get()  # mine
        for _r in range(below):  # the rest are for the rows below me
            io.w_out.put(io.w_in.get())
        for m in range(n):
            a = io.a_in.get()
            p = io.p_in.get()
            io.p_out.put(p + a * w)
            io.a_out.put(a)

    @spmw.fabric
    def g(A: int8[n, n], B: int8[n, n], C: int32[n, n]):
        P = spmw.place(mac, on=mesh)
        spmw.stream_in(B, into=P.w_in, index=(..., P.cols))  # column c: B[:, c]
        spmw.stream_in(A, into=P.a_in, index=(..., P.rows))  # row k: A[:, k]
        spmw.stream_in(0, into=P.p_in)  # every column's sum starts at 0
        spmw.gather(C, from_=P.p_out, index=(..., P.cols))  # column c: C[:, c]

    return g


gemm_ws = gemm_ws_of(SIZE)


def _operands(size=SIZE, seed=5):
    rng = np.random.default_rng(seed)
    a = rng.integers(-8, 8, size=(size, size)).astype(np.int8)
    b = rng.integers(-8, 8, size=(size, size)).astype(np.int8)
    return a, b, np.zeros((size, size), dtype=np.int32)


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_matches_numpy(target):
    a, b, c = _operands()
    spmw.build(gemm_ws, target=target)(a, b, c)
    np.testing.assert_array_equal(c, a.astype(np.int32) @ b.astype(np.int32))


def test_the_cell_carries_nothing():
    """No value survives an iteration, which is the point of the dataflow.

    The output-stationary cell accumulates (`test_spmw_gemm_int8.py` finds its
    `acc`), so its interval is its adder's latency.  This one adds into a value
    that arrived and leaves, so no adder -- integer or float -- sits in a cycle.
    """
    import ast

    from allo.spmw import schedule as sched
    from allo.spmw.role_ip import UnitEmitter

    emitter = UnitEmitter(spmw.elaborate(gemm_ws))
    placement = emitter.placements()[0]
    for order, (_sig, _routing, sites) in enumerate(emitter.classes(placement)):
        body, _pids, _rw = emitter.body_for(placement, order, sites[0])
        tree = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        assert sched.accumulators(tree) == []


def test_the_edge_is_three_bindings():
    """One stream a row, one a column, and one back: 3S, where the
    output-stationary mesh has a result to collect from every cell."""
    from allo.spmw import rtl

    graph = spmw.elaborate(gemm_ws)
    assert sorted(b.kind for b in graph.bindings) == [
        "gather",
        "seed",
        "stream_in",
        "stream_in",
    ]
    assert rtl.cost(graph)["streams"] == 3 * SIZE


@pytest.mark.parametrize("size", [2, 3, 8])
def test_it_scales(size):
    """The grid is a parameter, and the roles stop growing with it."""
    from allo.spmw import rtl

    cost = rtl.cost(spmw.elaborate(gemm_ws_of(size)))
    assert cost["instances"] == size * size
    if size >= 3:
        assert cost["roles"] == 9
    a, b, c = _operands(size=size)
    spmw.build(gemm_ws_of(size), target="ref")(a, b, c)
    np.testing.assert_array_equal(c, a.astype(np.int32) @ b.astype(np.int32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
