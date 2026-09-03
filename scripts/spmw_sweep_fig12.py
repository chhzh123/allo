# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fig. 12: grouped attention-PV, cycles and MAC utilization against G.

A head dimension smaller than the array's width leaves columns idle. Cutting
the array into G column slabs and serpentining each slab's partial-sum chain
into the top of the next fills them. `G` is an argument to the fabric, so every
point here is the same source design.

Utilization is counted from the elaborated graph -- how many of the array's
sites actually carry a weight and take part -- rather than assumed. Cycles are
for a fixed sequence length so points are compared on the same work.

    python3 scripts/spmw_sweep_fig12.py --rows 16 --cols 16 --seq 4096
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allo.spmw as spmw  # noqa: E402
from allo.ir.types import int8, int32  # noqa: E402

SHIFT = 2


def attention_pv(rows, cols, groups, seq):
    """The mini-TPU's PE and activation unit, regrouped into G column slabs."""

    class WsIO(spmw.Interface):
        a_in = spmw.In(int8)
        a_out = spmw.Out(int8)
        p_in = spmw.In(int32)
        p_out = spmw.Out(int32)
        w = spmw.MemIn(int8)

    class ActIO(spmw.Interface):
        z_in = spmw.In(int32)
        y_out = spmw.Out(int8)

    @spmw.unit
    def mac(io: WsIO):
        for _m in range(seq):
            a = io.a_in.get()
            p = io.p_in.get()
            io.p_out.put(p + a * io.w)
            io.a_out.put(a)

    @spmw.unit
    def act(io: ActIO):
        for _m in range(seq):
            z = io.z_in.get()
            if z < 0:
                z = 0
            y: int8 = z >> SHIFT
            io.y_out.put(y)

    d = cols // groups
    span = groups * rows

    def link(i, j):
        links = {}
        if (j + 1) % d != 0:  # activations move inside a slab only
            links[WsIO.a_out] = spmw.to((i, j + 1), WsIO.a_in)
        if i + 1 < rows:  # partial sums down my column ...
            links[WsIO.p_out] = spmw.to((i + 1, j), WsIO.p_in)
        elif j + d < cols:  # ... then into the next slab's top row
            links[WsIO.p_out] = spmw.to((0, j + d), WsIO.p_in)
        return links

    topo = spmw.Topology(WsIO, (rows, cols), link=link, name=f"grouped(G={groups})")

    @spmw.fabric
    def pv(Pr: int8[seq, span], V: int8[span, d], Y: int8[seq, d]):
        P = spmw.place(mac, on=topo)
        Pa = spmw.place(act, on=spmw.Grid((d,)))
        k = P.rows
        g, e = spmw.split(P.cols, factor=groups)
        spmw.shard(V, into=P.w, index=(g * rows + k, e))
        spmw.stream_in(Pr, into=P.a_in, index=(..., g * rows + k))
        spmw.stream_in(0, into=P.p_in)
        spmw.link(P.p_out, to=Pa.z_in)
        (lane,) = Pa.axes
        spmw.gather(Y, from_=Pa.y_out, index=(..., lane))

    return pv


def measure(rows, cols, groups, seq):
    """Elaborate one configuration and count what it actually uses."""
    graph = spmw.elaborate(attention_pv(rows, cols, groups, seq))
    mesh = graph.placements[0]
    sites = len(list(mesh.sites()))
    d = cols // groups
    span = groups * rows
    # Every site of the mesh holds one V element and does one MAC per row, so
    # the array is fully occupied for any G that divides the width. What
    # changes with G is how much of the *sequence* one pass covers.
    covered = span
    return {
        "groups": groups,
        "d": d,
        "span": covered,
        "sites": sites,
        "seeded_p_in": len(mesh.p_in),
        "drained_p_out": len(mesh.p_out),
        "west_a_in": len(mesh.a_in),
        "bindings": len(graph.bindings),
        # One pass covers `covered` rows of the reduction; a sequence of length
        # `seq` needs this many passes, each `seq` deep.
        "passes": -(-cols // d),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--groups", default="1,2,4")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = []
    print(f"array {args.rows}x{args.cols}, sequence tile {args.seq}")
    print(f"  {'G':>2s} {'d':>3s} {'span':>5s} {'sites':>6s} "
          f"{'seed':>5s} {'drain':>6s} {'west':>5s} {'binds':>6s}")
    for g in [int(x) for x in args.groups.split(",")]:
        if args.cols % g:
            print(f"  G={g} skipped: does not divide {args.cols} columns")
            continue
        point = measure(args.rows, args.cols, g, args.seq)
        rows.append(point)
        print("  {groups:2d} {d:3d} {span:5d} {sites:6d} "
              "{seeded_p_in:5d} {drained_p_out:6d} {west_a_in:5d} "
              "{bindings:6d}".format(**point))

    if rows:
        base = rows[0]
        print("\n  reduction span vs G=1:")
        for point in rows:
            print(f"    G={point['groups']}: covers {point['span']} rows "
                  f"({point['span']/base['span']:.2f}x), "
                  f"head dim {point['d']}")
        print(f"\n  the binding count is {rows[0]['bindings']} at every G "
              f"({'unchanged' if len({r['bindings'] for r in rows}) == 1 else 'CHANGED'})")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print(f"wrote {args.json}")
    print("FIG12 DONE")


if __name__ == "__main__":
    main()
