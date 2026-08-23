# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hierarchical tiled GEMM: compute and memory hierarchy read off the structure.

A fabric with a declared Interface is a composite component, so it is placed
directly -- the tile engine needs no wrapper unit.  What a placement exposes is
the same kind of thing a unit declares, which is why hierarchy needs no new
mechanism: expanding a placed fabric is just running its body once per site with
``io`` bound to the slice that site owns.
"""

import tempfile

import numpy as np
import pytest

import allo.backend.hls as hls
import allo.spmw as spmw
from allo.ir.types import float32

M, N, K = 4, 4, 4
Rt, Ct = 2, 2
TM, TN = M // Rt, N // Ct


class MacIO(spmw.Interface):
    west = spmw.In(float32)
    north = spmw.In(float32)
    east = spmw.Out(float32)
    south = spmw.Out(float32)
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


class TileIO(spmw.Interface):
    a = spmw.MemIn(float32[Rt, K])
    b = spmw.MemIn(float32[K, Ct])
    c = spmw.MemOut(float32[Rt, Ct])


@spmw.fabric(io=TileIO)
def tile_gemm(io: TileIO):
    """One tile engine, itself placeable."""
    P = spmw.place(pe, on=spmw.mesh(MacIO, (Rt, Ct)))
    spmw.stream_in(io.a, into=P.west, index=(P.rows, ...))
    spmw.stream_in(io.b, into=P.north, index=(..., P.cols))
    spmw.gather(io.c, from_=P.c)


@spmw.fabric
def tiled_gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
    T = spmw.place(tile_gemm, on=spmw.Grid((TM, TN)))
    spmw.shard(A, into=T.a, dim=0)  # host brick -> per-tile pieces
    spmw.shard(B, into=T.b, dim=1)
    spmw.shard(C, from_=T.c)  # identity blocks: tile (i, j) owns C[i, j]'s block


def _operands(seed=11):
    rng = np.random.default_rng(seed)
    return (
        rng.random((M, K), dtype=np.float32),
        rng.random((K, N), dtype=np.float32),
    )


def test_the_placed_fabric_expands():
    """The tile engine is written once and instantiated by placement."""
    graph = spmw.elaborate(tiled_gemm)
    live = [p for p in graph.placements if not p.expanded]
    gone = [p for p in graph.placements if p.expanded]
    assert len(gone) == 1  # the tile engine itself emits nothing
    assert len(live) == TM * TN  # one mesh per tile
    assert all(p.grid == (Rt, Ct) for p in live)


def test_each_tile_owns_its_slice():
    """`dim=` distributes a tensor axis along the grid axis of the same index."""
    graph = spmw.elaborate(tiled_gemm)
    shard_a, shard_b, shard_c = graph.bindings[:3]
    assert shard_a.imap.slice_for((0, 0)) == ((0, Rt), (0, K))
    assert shard_a.imap.slice_for((1, 0)) == ((Rt, Rt), (0, K))  # next tile row
    assert shard_b.imap.slice_for((0, 1)) == ((0, K), (Ct, Ct))  # next tile column
    assert shard_c.imap.slice_for((1, 1)) == ((Rt, Rt), (Ct, Ct))


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_matches_numpy(target):
    A, B = _operands()
    C = np.zeros((M, N), dtype=np.float32)
    spmw.build(tiled_gemm, target=target)(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)


@pytest.mark.skipif(not hls.is_available("vitis_hls"), reason="vitis_hls not on PATH")
def test_hls_csim_matches_numpy():
    """A placed fabric, expanded per tile, through the HLS path."""
    A, B = _operands()
    C = np.zeros((M, N), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        spmw.build(tiled_gemm, target="vitis_hls", mode="csim", project=tmpdir)(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)


def test_a_tile_must_be_fed():
    """A placed fabric's port with no binding has no data to work on."""

    @spmw.fabric
    def bad(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        T = spmw.place(tile_gemm, on=spmw.Grid((TM, TN)))
        spmw.shard(A, into=T.a, dim=0)
        spmw.shard(C, from_=T.c)  # b never bound

    with pytest.raises(spmw.SPMWBindingError, match="nothing binds it"):
        spmw.elaborate(bad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
