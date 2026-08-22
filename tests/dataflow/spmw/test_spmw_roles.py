# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Roles: variant bodies for sites whose behaviour genuinely differs.

Boundary differences that are only about *inputs* should be bindings, so the
interior body stays total.  A role is for the other case -- here the head of a
chain has no upstream to read and seeds the chain itself, which is behaviour, not
plumbing.
"""

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import float32

P = 5


class ChainIO(spmw.Interface):
    prev = spmw.In(float32)
    next = spmw.Out(float32)
    out = spmw.MemOut(float32)


@spmw.unit
def stage(io: ChainIO):
    v = io.prev.get()
    acc: float32 = v * 2.0
    io.next.put(acc)
    io.out = acc


@stage.role(unbound=(ChainIO.prev,))
def stage_head(io: ChainIO, site: spmw.Site):
    acc: float32 = 1.0
    io.next.put(acc)
    io.out = acc


@spmw.fabric
def chain(O: float32[P]):
    Q = spmw.place(
        stage,
        on=spmw.Topology(
            ChainIO,
            (P,),
            link=lambda i: {ChainIO.next: spmw.to((i + 1,), ChainIO.prev)},
        ),
    )
    spmw.gather(O, from_=Q.out)


EXPECTED = np.array([2.0**i for i in range(P)], dtype=np.float32)


def test_the_role_serves_only_the_head():
    graph = spmw.elaborate(chain)
    Q = graph.placements[0]
    assert Q.roles[(0,)].name == "stage_head"
    assert all(Q.roles[(i,)].name == "stage" for i in range(1, P))


def test_a_role_may_not_touch_its_unbound_ports():
    with pytest.raises(spmw.SPMWPlacementError, match="prev"):

        @stage.role(unbound=(ChainIO.prev,))
        def bad(io: ChainIO):
            x = io.prev.get()


def test_the_tail_elides_its_put():
    """The last stage's `next` has nowhere to go, so the put disappears."""
    text = spmw.source(chain)
    # Three arms: the head's role, the interior body, and the tail with no put.
    assert (
        text.count("meta_if") + text.count("meta_elif") + text.count("meta_else") == 3
    )


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_matches(target):
    O = np.zeros(P, dtype=np.float32)
    spmw.build(chain, target=target)(O)
    np.testing.assert_allclose(O, EXPECTED, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
