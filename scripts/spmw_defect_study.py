# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Table 4: seeded single-line structural defects, and who catches them.

Each mutation changes exactly one line of a working design. ``caught
statically`` means ``spmw.elaborate`` refuses it -- before HLS, before
simulation, before synthesis.

The controls matter as much as the mutations. Every design is elaborated
unmutated first, and a run that cannot elaborate its own controls is reported
as broken rather than as a perfect score: a checker that refuses everything
would otherwise look like a checker that catches everything.

    python3 scripts/spmw_defect_study.py [--json out.json]
"""

import argparse
import json
import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "tests", "dataflow", "spmw")
)

import allo.spmw as spmw
from allo.ir.types import int8, int32, float32


# --------------------------------------------------------------------------
# Design builders. Each takes a `bug` tag; `bug=None` builds the correct design.
# --------------------------------------------------------------------------

D = 4


def build_mesh(bug=None):
    """Systolic GEMM: a neighbour mesh with an edge feed and a drain."""

    class IO(spmw.Interface):
        a_in = spmw.In(int8)
        a_out = spmw.Out(int8)
        p_in = spmw.In(int32 if bug != "type_mismatch" else int8)
        p_out = spmw.Out(int32)

    def links(r, c):
        # off-by-one mutations shift exactly one subscript
        dr = 1 if bug != "offbyone_row" else 2
        dc = 1 if bug != "offbyone_col" else 2
        out = {
            IO.a_out: spmw.key(r, c + dc),
            IO.a_in: spmw.key(r, c),
            IO.p_out: spmw.key(r + dr, c, "p"),
            IO.p_in: spmw.key(r, c, "p"),
        }
        if bug == "dangling_writer":
            out[IO.a_out] = spmw.key(r, c + 99)
        if bug == "dangling_reader":
            out[IO.a_in] = spmw.key(r, c - 99)
        return out

    @spmw.unit
    def mac(io: IO):
        a = io.a_in.get()
        p = io.p_in.get()
        io.a_out.put(a)
        io.p_out.put(p + a * a)

    @spmw.fabric
    def fab(A: int8[D, D], Y: int32[D, D]):
        P = spmw.place(mac, on=spmw.Topology(IO, grid=(D, D), link=links))
        rows, cols = P.axes
        spmw.stream_in(A, into=P.a_in, index=(..., rows))
        if bug != "missing_seed":
            spmw.stream_in(0, into=P.p_in)
        idx = (..., cols) if bug != "gather_range" else (..., cols + D)
        spmw.gather(Y, from_=P.p_out, index=idx)

    return fab


def build_chain(bug=None):
    """Daisy-chain multi-cache GEMM: a linear chain with a boundary role."""

    class IO(spmw.Interface):
        x_in = spmw.In(int8)
        x_out = spmw.Out(int8)
        s_out = spmw.MemOut(int32)

    def links(i):
        step = 1 if bug != "offbyone_chain" else 2
        return {IO.x_out: spmw.key(i + step), IO.x_in: spmw.key(i)}

    @spmw.unit
    def stage(io: IO):
        x = io.x_in.get()
        io.x_out.put(x)
        io.s_out = x

    @spmw.fabric
    def fab(X: int8[D], S: int32[D]):
        P = spmw.place(stage, on=spmw.Topology(IO, grid=(D,), link=links))
        (i,) = P.axes
        spmw.stream_in(X, into=P.x_in, index=(i,))
        spmw.gather(S, from_=P.s_out, index=(i,))

    return fab


def build_staged(bug=None):
    """Weight-stationary MXU with a staged, phased weight tile."""

    class IO(spmw.Interface):
        a_in = spmw.In(int8)
        a_out = spmw.Out(int8)
        p_in = spmw.In(int32)
        p_out = spmw.Out(int32)
        w = spmw.MemIn(int8)

    def links(r, c):
        return {
            IO.a_out: spmw.key(r, c + 1),
            IO.a_in: spmw.key(r, c),
            IO.p_out: spmw.key(r + 1, c, "p"),
            IO.p_in: spmw.key(r, c, "p"),
        }

    @spmw.unit
    def mac(io: IO):
        a = io.a_in.get()
        p = io.p_in.get()
        io.a_out.put(a)
        io.p_out.put(p + a * io.w)

    @spmw.fabric
    def fab(A: int8[D, D], W: int8[D, D], Y: int32[D, D]):
        P = spmw.place(mac, on=spmw.Topology(IO, grid=(D, D), link=links))
        rows, cols = P.axes
        tile = spmw.mem(int8[D, D], layout=spmw.banked(on="col"), double=True)
        spmw.shard(tile, into=P.w)
        if bug == "unsync_two_writers":
            # two writers of the same brick inside one phase
            with spmw.phase("load"):
                spmw.copy(W, into=tile, how="dma")
                spmw.copy(W, into=tile, how="dma")
            with spmw.phase("compute"):
                spmw.stream_in(A, into=P.a_in, index=(..., rows))
                spmw.stream_in(0, into=P.p_in)
                spmw.gather(Y, from_=P.p_out, index=(..., cols))
        elif bug == "unsync_no_phase":
            # writer and readers share a phase: nothing orders them
            spmw.copy(W, into=tile, how="dma")
            spmw.stream_in(A, into=P.a_in, index=(..., rows))
            spmw.stream_in(0, into=P.p_in)
            spmw.gather(Y, from_=P.p_out, index=(..., cols))
        else:
            with spmw.phase("load"):
                spmw.copy(W, into=tile, how="dma")
            with spmw.phase("compute"):
                spmw.stream_in(A, into=P.a_in, index=(..., rows))
                spmw.stream_in(0, into=P.p_in)
                spmw.gather(Y, from_=P.p_out, index=(..., cols))

    return fab


def build_capacity(bug=None):
    """A design whose memory port capacity can be over-subscribed."""

    class IO(spmw.Interface):
        a_in = spmw.In(int8)
        a_out = spmw.Out(int8)
        w = spmw.MemIn(int8[D])
        s = spmw.MemOut(int32)

    def links(i):
        return {IO.a_out: spmw.key(i + 1), IO.a_in: spmw.key(i)}

    @spmw.unit
    def stage(io: IO):
        a = io.a_in.get()
        io.a_out.put(a)
        io.s = a * io.w[0]

    @spmw.fabric
    def fab(A: int8[D], W: int8[D, D], S: int32[D]):
        P = spmw.place(stage, on=spmw.Topology(IO, grid=(D,), link=links))
        (i,) = P.axes
        # the port holds D elements; a wider block over-subscribes it
        if bug == "capacity_block":
            spmw.shard(W, into=P.w, index=(i,), dim=0)
            spmw.shard(W, into=P.w, index=(i,), dim=0)
        elif bug == "capacity_shape":
            spmw.stationary(W, at=P.w)  # D*D into a D-element port
        else:
            spmw.shard(W, into=P.w, index=(i,), dim=0)
        spmw.stream_in(A, into=P.a_in, index=(i,))
        spmw.gather(S, from_=P.s, index=(i,))

    return fab


def build_boundary(bug=None):
    """A chain whose last site needs a different role."""

    class IO(spmw.Interface):
        x_in = spmw.In(int8)
        x_out = spmw.Out(int8)
        y = spmw.MemOut(int32)

    def links(i):
        out = {IO.x_in: spmw.key(i)}
        # the correct design stops the chain at the last site
        if bug == "missing_boundary" or i + 1 < D:
            out[IO.x_out] = spmw.key(i + 1)
        return out

    @spmw.unit
    def stage(io: IO):
        x = io.x_in.get()
        io.x_out.put(x)
        io.y = x

    @spmw.fabric
    def fab(X: int8[D], Y: int32[D]):
        P = spmw.place(stage, on=spmw.Topology(IO, grid=(D,), link=links))
        (i,) = P.axes
        spmw.stream_in(X, into=P.x_in, index=(i,))
        spmw.gather(Y, from_=P.y, index=(i,))

    return fab


# --------------------------------------------------------------------------
# The mutation table: (class, builder, bug tag, one-line description)
# --------------------------------------------------------------------------

MUTATIONS = [
    # 1. Mismatched channel endpoints
    (
        "Mismatched channel endpoints",
        build_mesh,
        "dangling_writer",
        "a_out names a key no site reads",
    ),
    (
        "Mismatched channel endpoints",
        build_mesh,
        "dangling_reader",
        "a_in names a key no site writes",
    ),
    (
        "Mismatched channel endpoints",
        build_mesh,
        "missing_seed",
        "p_in has no producer at the north edge",
    ),
    (
        "Mismatched channel endpoints",
        build_chain,
        "dangling_writer_chain",
        "chain writes past its last site",
    ),
    # 2. Off-by-one link arithmetic
    ("Off-by-one link arithmetic", build_mesh, "offbyone_row", "psum link skips a row"),
    (
        "Off-by-one link arithmetic",
        build_mesh,
        "offbyone_col",
        "activation link skips a column",
    ),
    (
        "Off-by-one link arithmetic",
        build_chain,
        "offbyone_chain",
        "chain step of 2 instead of 1",
    ),
    # 3. Missing boundary variant
    (
        "Missing boundary variant",
        build_boundary,
        "missing_boundary",
        "last site still drives x_out",
    ),
    # 4. Type mismatch across a channel
    (
        "Type mismatch across a channel",
        build_mesh,
        "type_mismatch",
        "int32 psum read through an int8 port",
    ),
    # 5. Unsynchronized shared memory
    (
        "Unsynchronized shared memory",
        build_staged,
        "unsync_two_writers",
        "two DMA writers of one brick in one phase",
    ),
    (
        "Unsynchronized shared memory",
        build_staged,
        "unsync_no_phase",
        "weight writer and readers share a phase",
    ),
    # 6. Port-capacity violation
    (
        "Port-capacity violation",
        build_capacity,
        "capacity_block",
        "two shards bind the same memory port",
    ),
    (
        "Port-capacity violation",
        build_capacity,
        "capacity_shape",
        "a D*D tensor made stationary on a D-element port",
    ),
    (
        "Port-capacity violation",
        build_mesh,
        "gather_range",
        "gather index runs past the tensor",
    ),
]

CONTROLS = [
    ("mesh", build_mesh),
    ("chain", build_chain),
    ("staged", build_staged),
    ("capacity", build_capacity),
    ("boundary", build_boundary),
]


def attempt(builder, bug):
    """Elaborate one design. Returns (ok, label)."""
    try:
        spmw.elaborate(builder(bug))
        return True, "elaborated"
    except Exception as exc:  # noqa: BLE001 - any refusal counts
        return False, type(exc).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the raw results here")
    args = ap.parse_args()

    print("=" * 66)
    print("controls (each must elaborate)")
    print("=" * 66)
    broken = []
    for name, builder in CONTROLS:
        ok, label = attempt(builder, None)
        print(f"  {name:12s} {'ok' if ok else 'BROKEN: ' + label}")
        if not ok:
            broken.append(name)

    print()
    print("=" * 66)
    print("mutations")
    print("=" * 66)
    rows = []
    for cls, builder, bug, desc in MUTATIONS:
        caught, label = attempt(builder, bug)
        caught = not caught  # elaboration refusing == caught
        rows.append(
            {
                "class": cls,
                "bug": bug,
                "desc": desc,
                "caught": caught,
                "error": label if caught else "",
            }
        )
        mark = "caught " + label if caught else "MISSED"
        print(f"  [{cls[:28]:28s}] {desc[:36]:36s} {mark}")

    print()
    print("=" * 66)
    print("Table 4")
    print("=" * 66)
    order = []
    for cls, _b, _t, _d in MUTATIONS:
        if cls not in order:
            order.append(cls)
    print(f"  {'Bug class':34s} {'Seeded':>7s} {'Caught':>7s}")
    tot_s = tot_c = 0
    for cls in order:
        sel = [r for r in rows if r["class"] == cls]
        c = sum(1 for r in sel if r["caught"])
        tot_s += len(sel)
        tot_c += c
        print(f"  {cls:34s} {len(sel):7d} {c:7d}")
    pct = (100.0 * tot_c / tot_s) if tot_s else 0.0
    print(f"  {'Total':34s} {tot_s:7d} {tot_c:7d}  ({pct:.0f}%)")

    if broken:
        print()
        print(f"WARNING: controls failed to elaborate: {', '.join(broken)}")
        print("The caught-rate above is not meaningful until they pass.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"rows": rows, "broken": broken}, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
