# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention's P.V on a fixed budget: two conventional passes against one grouped.

`test_spmw_attention.py` shows that grouping is a Python argument.  This file
asks the question a reviewer asks next: on the *same* 4x4 array, computing the
*same* product, what does grouping buy?

The product is O = P @ V with P [M, 8] and V [8, 2]: an output width d=2 on a
four-wide array, and a reduction of eight terms on a four-deep one.

* ``conventional`` keeps the plain mesh.  Activations forward east across all
  four columns, so every column sees the same four reduction terms and the two
  spare columns can hold nothing but padding.  Covering eight terms takes two
  passes over the same hardware, each with its own four-row weight tile, and
  the second pass has to pick up where the first left off: the partial leaves
  the array un-shifted and un-clipped (``drain`` taps the psum beside ``act``'s
  own output) and re-enters at the top of the columns on the next pass, where
  the psum chain -- already an adder chain -- adds the tile's terms onto it.
  Only the last pass's activation output is the result.

* ``grouped`` is ``attention_pv(2)`` from `test_spmw_attention.py`: the same
  sixteen PEs cut into two column slabs, slab 1 holding the second weight tile
  and the psum chain serpentining from slab 0's bottom into slab 1's top.  One
  pass covers all eight terms.

The PE is the paper's ``mac`` in both, unchanged.  The two fabrics differ in
the topology's group count, in the boundary bindings, and in the boundary
unit's tap; :func:`adaptation_diff` prints exactly that difference.
"""

import difflib
import functools
import inspect
import re
import tempfile
import textwrap

import numpy as np
import pytest

import allo.backend.hls as hls
import allo.spmw as spmw
from allo.ir.types import int8, int32
from test_spmw_attention import (
    C,
    MT,
    R,
    SHIFT,
    ActIO,
    WsIO,
    _reference,
    act as paper_act,
    grouped_mxu,
    mac as paper_mac,
)

D = 2  # the output width, held fixed
G = C // D  # column slabs the grouped design cuts the array into
L = G * R  # the reduction length: two four-deep tiles, or one pass of two slabs
PASSES = L // R  # what the conventional design needs to cover it
SEQS = (6, 64, 4096)
SEEDS = (0, 1, 2)


class DrainIO(spmw.Interface):
    z_in = spmw.In(int32)
    z_out = spmw.Out(int32)  # the sum as it is, so a later pass can resume it
    y_out = spmw.Out(int8)


def units(seq):
    """The paper's ``mac`` and ``act`` at sequence length ``seq``, and ``act`` with a tap.

    The trip count is the only free variable: `test_the_units_are_the_papers`
    checks the bodies against `test_spmw_attention.py` token for token.
    """

    @spmw.unit
    def mac(io: WsIO):
        for m in range(seq):
            a = io.a_in.get()
            p = io.p_in.get()
            io.p_out.put(p + a * io.w)
            io.a_out.put(a)

    @spmw.unit
    def act(io: ActIO):
        for m in range(seq):
            z = io.z_in.get()
            if z < 0:
                z = 0
            y: int8 = z >> SHIFT
            io.y_out.put(y)

    @spmw.unit
    def drain(io: DrainIO):
        for m in range(seq):
            z = io.z_in.get()
            io.z_out.put(z)
            if z < 0:
                z = 0
            y: int8 = z >> SHIFT
            io.y_out.put(y)

    return mac, act, drain


# -- the pair -----------------------------------------------------------------
#
# Two functions rather than one with a flag, so that the adaptation is a diff
# between them and nothing else: `adaptation_diff()` below.


def pv_conventional(seq):
    """The plain mesh, d=2 of its four columns useful, one four-row tile per pass."""
    mac, _act, drain = units(seq)

    @spmw.fabric
    def pv(
        Pr: int8[seq, R],
        V: int8[R, C],
        Zin: int32[seq, C],
        Zout: int32[seq, C],
        Y: int8[seq, C],
    ):
        P = spmw.place(mac, on=grouped_mxu(WsIO, (R, C), 1))
        Pa = spmw.place(drain, on=spmw.Grid((C,)))
        k, c = P.rows, P.cols
        spmw.shard(V, into=P.w, index=(k, c))  # PE (k, c) holds V[k, c]
        spmw.stream_in(Pr, into=P.a_in, index=(..., k))  # the west edge
        spmw.stream_in(Zin, into=P.p_in, index=(..., c))  # the top row resumes the sum
        spmw.link(P.p_out, to=Pa.z_in)  # the bottom row
        (lane,) = Pa.axes
        spmw.gather(Zout, from_=Pa.z_out, index=(..., lane))
        spmw.gather(Y, from_=Pa.y_out, index=(..., lane))

    return pv


def pv_grouped(seq):
    """Two column slabs, the psum chain serpentining between them: one pass."""
    mac, act, _drain = units(seq)

    @spmw.fabric
    def pv(
        Pr: int8[seq, L],
        V: int8[L, D],
        Y: int8[seq, D],
    ):
        P = spmw.place(mac, on=grouped_mxu(WsIO, (R, C), G))
        Pa = spmw.place(act, on=spmw.Grid((D,)))
        k = P.rows
        g, e = spmw.split(P.cols, factor=G)  # g: which slab; e: column in it
        spmw.shard(V, into=P.w, index=(g * R + k, e))  # PE (k, c) holds V[g.R+k, e]
        spmw.stream_in(Pr, into=P.a_in, index=(..., g * R + k))  # each slab's west edge
        spmw.stream_in(0, into=P.p_in)  # slab 0's top row
        spmw.link(P.p_out, to=Pa.z_in)  # the last slab's bottom row
        (lane,) = Pa.axes
        spmw.gather(Y, from_=Pa.y_out, index=(..., lane))

    return pv


DESIGNS = {"conventional": pv_conventional, "grouped": pv_grouped}


def adaptation_diff(context=1):
    """The unified diff between the two fabrics -- the adaptation, exactly."""
    before = inspect.getsource(pv_conventional).splitlines()
    after = inspect.getsource(pv_grouped).splitlines()
    return "\n".join(
        difflib.unified_diff(
            before, after, "pv_conventional", "pv_grouped", n=context, lineterm=""
        )
    )


# -- running them --------------------------------------------------------------


def operands(seq, seed):
    """P [seq, L] and V [L, D], small signed integers so int8 results never wrap."""
    rng = np.random.default_rng(seed)
    P = rng.integers(-4, 4, (seq, L)).astype(np.int8)
    V = rng.integers(-4, 4, (L, D)).astype(np.int8)
    return P, V


def conventional_pass(t, P, V, Z):
    """The conventional design's operands for pass ``t``: its slice of P, its
    tile of V zero-padded to the array's width, and the partial to resume."""
    seq = P.shape[0]
    Pr = np.ascontiguousarray(P[:, t * R : (t + 1) * R])
    Vt = np.zeros((R, C), dtype=np.int8)
    Vt[:, :D] = V[t * R : (t + 1) * R, :]
    Zin = np.ascontiguousarray(Z, dtype=np.int32)
    Zout = np.zeros((seq, C), dtype=np.int32)
    Y = np.zeros((seq, C), dtype=np.int8)
    return Pr, Vt, Zin, Zout, Y


def run_conventional(launch, P, V):
    """Two launches of the one conventional fabric, the second seeded with the
    first's partial.  Returns the result and the partial that crossed over."""
    Z = np.zeros((P.shape[0], C), dtype=np.int32)
    for t in range(PASSES):
        Pr, Vt, Zin, Zout, Y = conventional_pass(t, P, V, Z)
        launch(Pr, Vt, Zin, Zout, Y)
        Z = Zout
    return Y[:, :D].copy(), Z


def run_grouped(launch, P, V):
    Y = np.zeros((P.shape[0], D), dtype=np.int8)
    launch(P, V, Y)
    return Y


def run(design, launch, P, V):
    if design == "conventional":
        return run_conventional(launch, P, V)[0]
    return run_grouped(launch, P, V)


@functools.lru_cache(maxsize=None)
def _launch(design, seq, target):
    """Built once per (design, size, target); the seeds share it."""
    return spmw.build(DESIGNS[design](seq), target=target)


# -- tests ---------------------------------------------------------------------


def test_the_units_are_the_papers():
    """The bodies are `test_spmw_attention.py`'s; only the trip count is free."""

    def body(unit):
        return textwrap.dedent(inspect.getsource(unit.fn)).replace("range(MT)", "range(seq)")

    mac, act, drain = units(MT)
    assert body(mac) == body(paper_mac)
    assert body(act) == body(paper_act)
    # `drain` is `act` plus the tap, and the tap is the only added line.
    added = [
        line[1:].strip()
        for line in difflib.unified_diff(
            body(act).splitlines(), body(drain).splitlines(), n=0, lineterm=""
        )
        if line.startswith("+") and not line.startswith("+++")
    ]
    assert added == ["def drain(io: DrainIO):", "io.z_out.put(z)"]


def test_the_same_array_underneath():
    """Sixteen PEs in both; what differs is which of their ports the topology leaves open."""
    conv = spmw.elaborate(pv_conventional(MT)).placements[0]
    grp = spmw.elaborate(pv_grouped(MT)).placements[0]
    assert len(list(conv.sites())) == len(list(grp.sites())) == R * C
    assert (len(conv.a_in), len(conv.p_in), len(conv.p_out)) == (R, C, C)
    assert (len(grp.a_in), len(grp.p_in), len(grp.p_out)) == (R * G, D, D)


def changed_lines():
    """The adaptation's changed lines, each classified by what kind of line it is.

    A line is a tensor signature, a placement, a binding, an axis symbol, the
    choice of boundary unit, or the function's own header and docstring -- or
    it is something else, which is what the test below forbids: nothing in the
    diff may be arithmetic.
    """
    kinds = (
        ("signature", r"^\w+: int(8|32)\["),
        ("placement", r"^\w+ = spmw\.place\("),
        ("binding", r"^spmw\.(shard|stream_in|link|gather)\("),
        ("axes", r"^(\(lane,\)|k, c|k|g, e) = "),
        ("unit-choice", r"^mac, .*= units\(seq\)$"),
        ("header", r'^(def pv_|def pv\(|"""|\):$|return pv$)'),
    )
    out = []
    for line in adaptation_diff(context=0).splitlines():
        if line[:1] not in "+-" or line.startswith(("+++", "---")):
            continue
        text = line[1:].strip()
        kind = next((k for k, pat in kinds if re.match(pat, text)), "other")
        out.append((line[0], kind, text))
    return out


def test_the_adaptation_is_topology_and_bindings():
    """Every changed line is a signature, a placement, a binding or an axis; none is arithmetic."""
    changed = changed_lines()
    assert changed, "the two fabrics must differ"
    others = [text for _sign, kind, text in changed if kind == "other"]
    assert not others, f"lines that are neither topology nor binding: {others}"
    assert any(kind == "placement" for _s, kind, _t in changed)  # the group count
    assert any(kind == "binding" for _s, kind, _t in changed)  # the loaders and drains


@pytest.mark.parametrize("seq", SEQS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("design", sorted(DESIGNS))
def test_reference_matches(design, seq, seed):
    P, V = operands(seq, seed)
    Y = run(design, _launch(design, seq, "ref"), P, V)
    np.testing.assert_array_equal(Y, _reference(P, V))


@pytest.mark.parametrize("seq", SEQS)
@pytest.mark.parametrize("seed", SEEDS)
def test_the_passes_hand_over_the_raw_sum(seq, seed):
    """What crosses between the passes is the un-shifted, un-clipped int32 partial."""
    P, V = operands(seq, seed)
    Y, Z = run_conventional(_launch("conventional", seq, "ref"), P, V)
    full = P.astype(np.int32) @ V.astype(np.int32)
    np.testing.assert_array_equal(Z[:, :D], full)  # the final partial is the whole sum
    np.testing.assert_array_equal(Z[:, D:], 0)  # the padding columns carry nothing
    np.testing.assert_array_equal(Y, _reference(P, V))


@pytest.mark.parametrize("seq", SEQS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("design", sorted(DESIGNS))
def test_simulator_matches(design, seq, seed):
    P, V = operands(seq, seed)
    Y = run(design, _launch(design, seq, "simulator"), P, V)
    np.testing.assert_array_equal(Y, _reference(P, V))


@pytest.mark.parametrize("seq", SEQS)
@pytest.mark.parametrize("seed", SEEDS)
def test_both_designs_agree_bit_for_bit(seq, seed):
    P, V = operands(seq, seed)
    conv = run("conventional", _launch("conventional", seq, "ref"), P, V)
    grp = run("grouped", _launch("grouped", seq, "ref"), P, V)
    np.testing.assert_array_equal(conv, grp)


@pytest.mark.skipif(not hls.is_available("vitis_hls"), reason="vitis_hls not on PATH")
@pytest.mark.parametrize("design", sorted(DESIGNS))
def test_hls_csim_matches(design):
    P, V = operands(MT, SEEDS[0])
    with tempfile.TemporaryDirectory() as tmpdir:
        launch = spmw.build(
            DESIGNS[design](MT), target="vitis_hls", mode="csim", project=tmpdir
        )
        Y = run(design, launch, P, V)
    np.testing.assert_array_equal(Y, _reference(P, V))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
