# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import allo.spmw as spmw
from allo.ir.types import float32, int32


def _systolic_twin(M, N, K):
    grid = spmw.mesh((M, N))

    @spmw.unit
    def pe(ctx):
        c: float32 = 0
        for k in range(K):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    return gemm


def test_declared_fifo_depth_reaches_dataflow_source():
    """A `depths=` on the systolic map is threaded into the generated dataflow FIFO arrays (previously
    hard-coded to 4), so the simulator/HLS dataflow backend buffers exactly what the SPMW program
    declared -- matching the rolled paths and the strict depth-consistency checker."""
    # pylint: disable=import-outside-toplevel
    from allo.spmw import _collect, _validate_collection
    from allo.spmw_datapath import generate_source

    grid = spmw.mesh((2, 2))

    @spmw.unit
    def pe(ctx):
        c: float32 = 0
        for k in range(2):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: float32[2, 2], B: float32[2, 2], C: float32[2, 2]):
        spmw.map(pe, grid=grid, depths={"east": 8, "west": 8, "north": 6, "south": 6})
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    src = generate_source(
        gemm, _validate_collection(_collect(gemm), strict_topology=True)
    )
    assert (
        "fifo_A: Stream[float32, 8]" in src
    )  # A family (west/east) honors the declared depth 8
    assert (
        "fifo_B: Stream[float32, 6]" in src
    )  # B family (north/south) honors the declared depth 6


def test_systolic_twin_runs_on_simulator():
    M, N, K = 2, 2, 2
    module = spmw.build(_systolic_twin(M, N, K), target="simulator")
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    module(A, B, C)
    np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)


def _systolic_twin_int32(M, N, K):
    grid = spmw.mesh((M, N))

    @spmw.unit
    def pe(ctx):
        c: int32 = 0
        for k in range(K):
            a: int32 = ctx.west.get()
            b: int32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: int32[M, K], B: int32[K, N], C: int32[M, N]):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    return gemm


@pytest.mark.parametrize("M, N, K", [(4, 4, 4), (3, 4, 5)])
def test_systolic_runs_for_arbitrary_sizes(M, N, K):
    module = spmw.build(_systolic_twin(M, N, K), target="simulator")
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    module(A, B, C)
    np.testing.assert_allclose(C, np.dot(A, B), atol=1e-3)


def test_systolic_runs_for_int32():
    M, N, K = 3, 3, 3
    module = spmw.build(_systolic_twin_int32(M, N, K), target="simulator")
    A = np.random.randint(0, 5, (M, K)).astype(np.int32)
    B = np.random.randint(0, 5, (K, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    module(A, B, C)
    np.testing.assert_array_equal(C, A @ B)


def test_systolic_non_canonical_operand_order_rejected():
    """The systolic desugar + rolled lowering require the W->E operand to be region operand #0 and the
    N->S operand #1 (the halo loaders resolve by tensor name while the role ABI and spmw.map operands
    are positional; they only agree in canonical order). A non-canonical `(B, A, C)` declaration -- A is
    the W->E operand but is declared second -- is rejected at build rather than mis-lowered.
    """
    M, N, K = 3, 4, 2
    grid = spmw.mesh((M, N))

    @spmw.unit
    def pe(ctx):
        c: float32 = 0
        for k in range(K):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(
        B: float32[K, N], A: float32[M, K], C: float32[M, N]
    ):  # non-canonical: A (W->E) is declared second
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    with pytest.raises(spmw.SPMWError, match="canonical order"):
        spmw.build(gemm, target="simulator")
    with pytest.raises(spmw.SPMWError, match="canonical order"):
        spmw.lower(gemm)  # the rolled path rejects it too


def test_grid_tensor_mismatch_rejected():
    # a declared mesh that disagrees with the operand-derived M x N grid must not silently compile
    M, N, K = 3, 3, 3
    grid = spmw.mesh((M + 1, N))

    @spmw.unit
    def pe(ctx):
        c: float32 = 0
        for k in range(K):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    with pytest.raises(spmw.SPMWError, match="grid"):
        spmw.build(gemm, target="simulator")


def test_non_systolic_region_rejected():
    grid = spmw.mesh((3, 3))

    @spmw.unit
    def pe(ctx):
        ctx.west.get()  # consumes `west`, never relays it, no stream_in -> boundary dangles

    @spmw.region()
    def r(A: float32[3, 3]):
        spmw.map(pe, grid=grid)  # not the systolic pattern: `west` has no data source

    # The strict topology check rejects the dangling boundary before the datapath recognizer is reached,
    # giving the precise root cause (the unhandled `west`) rather than a generic non-systolic error.
    with pytest.raises(spmw.SPMWError, match="unhandled"):
        spmw.build(r, target="simulator")
