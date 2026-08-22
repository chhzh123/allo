# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Systolic GEMM written in SPMW, checked against numpy and a dataflow oracle."""

import numpy as np
import pytest

import allo
import allo.dataflow as df
import allo.spmw as spmw
from allo.ir.types import Stream, float32

M, N, K = 4, 4, 4


class NSEW(spmw.Interface):
    west = spmw.In(float32)
    north = spmw.In(float32)
    east = spmw.Out(float32)
    south = spmw.Out(float32)


class MacIO(NSEW):
    c = spmw.MemOut(float32)


@spmw.unit
def pe(io: MacIO):
    acc: float32 = 0
    for k in range(K):
        a = io.west.get()
        b = io.north.get()
        acc += a * b
        io.east.put(a)
        io.south.put(b)
    io.c = acc


@spmw.fabric
def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    P = spmw.place(pe, on=spmw.mesh(MacIO, (M, N)))
    spmw.stream_in(A, into=P.west, index=(P.rows, ...))
    spmw.stream_in(B, into=P.north, index=(..., P.cols))
    spmw.gather(C, from_=P.c)


# The hand-written dataflow program the SPMW twin must agree with, bit for bit.
P0, P1 = M + 2, N + 2


@df.region()
def oracle(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    fifo_A: Stream[float32, 2][P0, P1]
    fifo_B: Stream[float32, 2][P0, P1]

    @df.kernel(mapping=[P0, P1], args=[A, B, C])
    def body(local_A: float32[M, K], local_B: float32[K, N], local_C: float32[M, N]):
        i, j = df.get_pid()
        with allo.meta_if(i in {0, M + 1} and j in {0, N + 1}):
            pass
        with allo.meta_elif(j == 0):
            for k in range(K):
                fifo_A[i, j + 1].put(local_A[i - 1, k])
        with allo.meta_elif(i == 0):
            for k in range(K):
                fifo_B[i + 1, j].put(local_B[k, j - 1])
        with allo.meta_elif(i == M + 1 and j > 0):
            for k in range(K):
                b: float32 = fifo_B[i, j].get()
        with allo.meta_elif(j == N + 1 and i > 0):
            for k in range(K):
                a: float32 = fifo_A[i, j].get()
        with allo.meta_else():
            c: float32 = 0
            for k in range(K):
                a: float32 = fifo_A[i, j].get()
                b: float32 = fifo_B[i, j].get()
                c += a * b
                fifo_A[i, j + 1].put(a)
                fifo_B[i + 1, j].put(b)
            local_C[i - 1, j - 1] = c


def _operands(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)
    return A, B


def test_elaborates():
    graph = spmw.elaborate(gemm)
    assert len(graph.placements) == 1
    assert [b.kind for b in graph.bindings] == ["stream_in", "stream_in", "gather"]
    placement = graph.placements[0]
    # Interior, four edges, four corners -- nine, and nine at any size.
    assert len(placement.topology.signatures()) == 9
    # Every rim input is a binding, so one total body serves every site.
    assert len(placement.role_classes()) == 1


def test_signature_count_is_size_independent():
    for size in (4, 8, 16):
        topo = spmw.mesh(MacIO, (size, size))
        assert len(topo.signatures()) == 9


def test_reference_matches_numpy():
    """The graph run directly -- one task per site, no compiler involved."""
    A, B = _operands()
    C = np.zeros((M, N), dtype=np.float32)
    spmw.build(gemm, target="ref")(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)


def test_simulator_matches_numpy():
    A, B = _operands()
    C = np.zeros((M, N), dtype=np.float32)
    mod = spmw.build(gemm, target="simulator")
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)


def test_bit_identical_to_dataflow_oracle():
    A, B = _operands(seed=1)
    got = np.zeros((M, N), dtype=np.float32)
    want = np.zeros((M, N), dtype=np.float32)

    spmw.build(gemm, target="simulator")(A, B, got)
    df.build(oracle, target="simulator")(A, B, want)
    # Not allclose: the transcribed body performs the same operations in the
    # same order, so the results must agree exactly.
    np.testing.assert_array_equal(got, want)


def test_lowered_program_is_rolled():
    """The emitted program has one body per signature class, not one per site."""
    text = spmw.source(gemm)
    assert text.count("df.kernel") == 3  # the array plus two loaders
    arms = text.count("meta_if") + text.count("meta_elif") + text.count("meta_else")
    assert arms <= 9, f"expected at most one arm per signature class, got {arms}"


def _gemm_of(size):
    """A GEMM fabric at a chosen size -- the grid is a parameter, not structure."""

    class IO(spmw.Interface):
        west = spmw.In(float32)
        north = spmw.In(float32)
        east = spmw.Out(float32)
        south = spmw.Out(float32)
        c = spmw.MemOut(float32)

    @spmw.unit
    def cell(io: IO):
        acc: float32 = 0
        for k in range(size):
            a = io.west.get()
            b = io.north.get()
            acc += a * b
            io.east.put(a)
            io.south.put(b)
        io.c = acc

    @spmw.fabric
    def small(A: float32[size, size], B: float32[size, size], C: float32[size, size]):
        P = spmw.place(cell, on=spmw.mesh(IO, (size, size)))
        spmw.stream_in(A, into=P.west, index=(P.rows, ...))
        spmw.stream_in(B, into=P.north, index=(..., P.cols))
        spmw.gather(C, from_=P.c)

    return small


@pytest.mark.parametrize("size", [2, 3, 5])
def test_other_sizes(size):
    """Resizing the array is a constant change, not a structural rewrite."""
    rng = np.random.default_rng(size)
    A = rng.random((size, size), dtype=np.float32)
    B = rng.random((size, size), dtype=np.float32)
    C = np.zeros((size, size), dtype=np.float32)
    spmw.build(_gemm_of(size), target="simulator")(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
