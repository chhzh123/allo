# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The systolic GEMM again, accumulating in integers.

Structurally identical to §3.1: the same mesh, the same computed boundaries, the
same loaders and drain, the same output-stationary body holding a result across
the whole `k` loop.  Only the arithmetic differs -- int8 operands into an int32
accumulator instead of float32 throughout.

That one change is the difference between II=7 and II=1.  The float version's
recurrence runs `acc -> fadd -> acc` with a dependence distance of 1, so its
interval is the adder's latency; an integer add is single-cycle, so the same
recurrence costs nothing.  It is not the systolic structure that stops the float
mesh reaching peak, and this design is the control that shows it.
"""

import tempfile

import numpy as np
import pytest

import allo.backend.hls as hls
import allo.spmw as spmw
from allo.ir.types import int8, int32

N = 4


class MacIO(spmw.Interface):
    west = spmw.In(int8)
    north = spmw.In(int8)
    east = spmw.Out(int8)
    south = spmw.Out(int8)
    c = spmw.MemOut(int32)  # the product of two int8s, summed N times


@spmw.unit
def pe(io: MacIO):
    acc: int32 = 0
    for k in range(N):
        a = io.west.get()
        b = io.north.get()
        acc += a * b
        io.east.put(a)
        io.south.put(b)
    io.c = acc


@spmw.fabric
def gemm_int8(A: int8[N, N], B: int8[N, N], C: int32[N, N]):
    P = spmw.place(pe, on=spmw.mesh(MacIO, (N, N)))
    spmw.stream_in(A, into=P.west, index=(P.rows, ...))
    spmw.stream_in(B, into=P.north, index=(..., P.cols))
    spmw.gather(C, from_=P.c)


def _operands(seed=3):
    """Small enough that the int32 accumulator cannot overflow."""
    rng = np.random.default_rng(seed)
    a = rng.integers(-8, 8, size=(N, N)).astype(np.int8)
    b = rng.integers(-8, 8, size=(N, N)).astype(np.int8)
    return a, b, np.zeros((N, N), dtype=np.int32)


def test_reference_matches_numpy():
    a, b, c = _operands()
    spmw.build(gemm_int8, target="ref")(a, b, c)
    np.testing.assert_array_equal(c, a.astype(np.int32) @ b.astype(np.int32))


def test_simulator_matches_numpy():
    a, b, c = _operands()
    spmw.build(gemm_int8, target="simulator")(a, b, c)
    np.testing.assert_array_equal(c, a.astype(np.int32) @ b.astype(np.int32))


def test_it_is_the_same_shape_as_the_float_mesh():
    """Nine roles on a 4x4 mesh -- interior, four edges, four corners.

    If this differed from the float version the comparison would not be a
    control, so it is checked rather than assumed.
    """
    from allo.spmw import rtl

    from test_spmw_rolled import gemm_of

    integer = rtl.cost(spmw.elaborate(gemm_int8))
    floating = rtl.cost(spmw.elaborate(gemm_of(N)))
    assert integer["roles"] == floating["roles"] == 9
    assert integer["instances"] == floating["instances"] == N * N


def test_it_still_carries_an_accumulator():
    """The recurrence is there; it is the adder underneath that is cheap.

    An integer mesh reaching II=1 while carrying a value is exactly what shows
    the interval is set by the operation in the cycle, not by the cycle.
    """
    import ast

    from allo.spmw import schedule as sched
    from allo.spmw.role_ip import UnitEmitter

    emitter = UnitEmitter(spmw.elaborate(gemm_int8))
    placement = emitter.placements()[0]
    _sig, _routing, sites = emitter.classes(placement)[2]
    body, _pids, _rw = emitter.body_for(placement, 2, sites[0])
    tree = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    assert sched.accumulators(tree) == ["acc"]


@pytest.mark.skipif(not hls.is_available("vitis_hls"), reason="vitis_hls not found")
def test_hls_csim_matches_numpy():
    a, b, c = _operands()
    with tempfile.TemporaryDirectory() as tmpdir:
        spmw.build(gemm_int8, target="vitis_hls", mode="csim", project=tmpdir)(a, b, c)
    np.testing.assert_array_equal(c, a.astype(np.int32) @ b.astype(np.int32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
