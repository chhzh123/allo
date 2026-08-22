# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The mini-TPU: a weight-stationary MXU with a fused activation row.

Activations march east, partial sums march south into an accumulator row.  The
interior body is total -- the top row's missing upstream is a scalar binding and
the chain's end is wired to a second placement -- so nothing is hand-specialised.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int8, int32

KT, NT, MT = 4, 4, 6
SHIFT = 4


class WsIO(spmw.Interface):
    a_in = spmw.In(int8)
    a_out = spmw.Out(int8)
    p_in = spmw.In(int32)
    p_out = spmw.Out(int32)
    w = spmw.MemIn(int8)  # my stationary weight: rank-0, both axes distributed


class ActIO(spmw.Interface):
    z_in = spmw.In(int32)
    y_out = spmw.Out(int8)


@spmw.unit
def mac(io: WsIO):
    for m in range(MT):
        a = io.a_in.get()
        p = io.p_in.get()
        io.p_out.put(p + a * io.w)
        io.a_out.put(a)


@spmw.unit
def act(io: ActIO):
    for m in range(MT):
        z = io.z_in.get()
        if z < 0:
            z = 0
        y: int8 = z >> SHIFT
        io.y_out.put(y)


def mxu_links(i, j):
    return {
        WsIO.a_out: spmw.to((i, j + 1), WsIO.a_in),
        WsIO.p_out: spmw.to((i + 1, j), WsIO.p_in),
    }


mxu = spmw.Topology(WsIO, grid=(KT, NT), link=mxu_links)


@spmw.fabric
def tpu_matmul(A: int8[MT, KT], W: int8[KT, NT], Y: int8[MT, NT]):
    P = spmw.place(mac, on=mxu)
    Pact = spmw.place(act, on=spmw.Grid((NT,)))
    spmw.shard(W, into=P.w)  # no index=: site (k, c) holds W[k, c]
    spmw.stream_in(A, into=P.a_in, index=(..., P.rows))  # row k pulls A[:, k]
    spmw.stream_in(0, into=P.p_in)  # north edge: seed every column at 0
    spmw.link(P.p_out, to=Pact.z_in)  # south edge -> the activation row
    (lane,) = Pact.axes
    spmw.gather(Y, from_=Pact.y_out, index=(..., lane))


def _reference(A, W):
    z = A.astype(np.int32) @ W.astype(np.int32)
    return (np.maximum(z, 0) >> SHIFT).astype(np.int8)


def _operands(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.integers(-8, 8, (MT, KT), dtype=np.int8)
    W = rng.integers(-8, 8, (KT, NT), dtype=np.int8)
    return A, W


def test_elaborates():
    graph = spmw.elaborate(tpu_matmul)
    assert [b.kind for b in graph.bindings] == [
        "shard",
        "stream_in",
        "seed",
        "link",
        "gather",
    ]
    assert len(graph.placements) == 2
    mxu_p, act_p = graph.placements
    # The chain is seeded at the top row and drained at the bottom, both of
    # which the link rule leaves open.
    assert len(mxu_p.p_in) == NT
    assert len(mxu_p.p_out) == NT
    assert len(act_p.z_in) == NT


def test_seed_folds_into_the_body():
    """A rank-0 source needs no brick and no loader; it folds into the site."""
    text = spmw.source(tpu_matmul)
    assert "p_in_load" not in text
    assert "p = 0" in text


def test_reference_matches():
    A, W = _operands()
    Y = np.zeros((MT, NT), dtype=np.int8)
    spmw.build(tpu_matmul, target="ref")(A, W, Y)
    np.testing.assert_array_equal(Y, _reference(A, W))


def test_simulator_matches_reference():
    A, W = _operands()
    Y = np.zeros((MT, NT), dtype=np.int8)
    spmw.build(tpu_matmul, target="simulator")(A, W, Y)
    np.testing.assert_array_equal(Y, _reference(A, W))


def test_weights_are_stationary():
    """The weight port is rank-0: each site owns exactly one element of W."""
    graph = spmw.elaborate(tpu_matmul)
    shard = graph.bindings[0]
    assert shard.kind == "shard"
    assert shard.target.port is WsIO.w
    assert WsIO.w.shape == ()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_repeated_operands(seed):
    A, W = _operands(seed)
    Y = np.zeros((MT, NT), dtype=np.int8)
    spmw.build(tpu_matmul, target="simulator")(A, W, Y)
    np.testing.assert_array_equal(Y, _reference(A, W))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
