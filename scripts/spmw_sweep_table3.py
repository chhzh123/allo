# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Table 3: architectural parameter sweeps driven from one source design.

Each point is elaborated and synthesised independently. Area is the sum over
roles of that role's HLS estimate times the number of sites it covers, which is
what the split backend actually instantiates; it is an HLS estimate and is
labelled as such, not a post-route number.

Latency is cycles for a fixed 256x256x256 problem tiled onto the array, so
points with different shapes are compared on the same work.

    python3 scripts/spmw_sweep_table3.py --out /scratch/$USER/table3
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "dataflow", "spmw"),
)

from spmw_build_array import stage, synthesise  # noqa: E402
from allo.spmw.role_ip import UnitEmitter  # noqa: E402
import allo.spmw as spmw  # noqa: E402
from allo.ir.types import int8, int32  # noqa: E402

PROBLEM = 256  # the fixed GEMM every point is measured on


def gemm_mesh(rows, cols, depth):
    """A rows x cols output-stationary mesh whose K extent is `depth`."""

    class MacIO(spmw.Interface):
        west = spmw.In(int8)
        north = spmw.In(int8)
        east = spmw.Out(int8)
        south = spmw.Out(int8)
        c = spmw.MemOut(int32)

    @spmw.unit
    def pe(io: MacIO):
        acc: int32 = 0
        for _k in range(depth):
            a = io.west.get()
            b = io.north.get()
            acc += a * b
            io.east.put(a)
            io.south.put(b)
        io.c = acc

    @spmw.fabric
    def fab(A: int8[rows, depth], B: int8[depth, cols], C: int32[rows, cols]):
        P = spmw.place(pe, on=spmw.mesh(MacIO, (rows, cols)))
        spmw.stream_in(A, into=P.west, index=(P.rows, ...))
        spmw.stream_in(B, into=P.north, index=(..., P.cols))
        spmw.gather(C, from_=P.c)

    return fab



def gemm_tiled(tiles, pe, depth):
    """A tiles x tiles grid of pe x pe engines: the hierarchical GEMM."""
    side = tiles * pe

    class MacIO(spmw.Interface):
        west = spmw.In(int8)
        north = spmw.In(int8)
        east = spmw.Out(int8)
        south = spmw.Out(int8)
        c = spmw.MemOut(int32)

    @spmw.unit
    def cell(io: MacIO):
        acc: int32 = 0
        for _k in range(depth):
            a = io.west.get()
            b = io.north.get()
            acc += a * b
            io.east.put(a)
            io.south.put(b)
        io.c = acc

    class TileIO(spmw.Interface):
        a = spmw.MemIn(int8[pe, depth])
        b = spmw.MemIn(int8[depth, pe])
        c = spmw.MemOut(int32[pe, pe])

    @spmw.fabric(io=TileIO)
    def engine(io: TileIO):
        P = spmw.place(cell, on=spmw.mesh(MacIO, (pe, pe)))
        spmw.stream_in(io.a, into=P.west, index=(P.rows, ...))
        spmw.stream_in(io.b, into=P.north, index=(..., P.cols))
        spmw.gather(io.c, from_=P.c)

    @spmw.fabric
    def fab(A: int8[side, depth], B: int8[depth, side], C: int32[side, side]):
        T = spmw.place(engine, on=spmw.Grid((tiles, tiles)))
        spmw.shard(A, into=T.a, dim=0)
        spmw.shard(B, into=T.b, dim=1)
        spmw.shard(C, from_=T.c)

    return fab


def attention_pv(rows, cols, groups, seq):
    """The grouped attention-PV array: G column slabs, chained psums."""

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
            y: int8 = z >> 2
            io.y_out.put(y)

    d = cols // groups
    span = groups * rows

    def link(i, j):
        links = {}
        if (j + 1) % d != 0:
            links[WsIO.a_out] = spmw.to((i, j + 1), WsIO.a_in)
        if i + 1 < rows:
            links[WsIO.p_out] = spmw.to((i + 1, j), WsIO.p_in)
        elif j + d < cols:
            links[WsIO.p_out] = spmw.to((0, j + d), WsIO.p_in)
        return links

    topo = spmw.Topology(WsIO, (rows, cols), link=link, name=f"grouped(G={groups})")

    @spmw.fabric
    def fab(Pr: int8[seq, span], V: int8[span, d], Y: int8[seq, d]):
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

    return fab


def points_for(family, dims, depth):
    """(tag, fabric, cycles) for every configuration of one family."""
    out = []
    if family == "mesh":
        for rows in dims:
            for cols in dims:
                out.append(
                    (f"mesh_{rows}x{cols}", gemm_mesh(rows, cols, depth),
                     cycles_for(rows, cols, depth))
                )
    elif family == "tiled":
        for tiles in (2, 4):
            for pe in (2, 4, 8):
                side = tiles * pe
                out.append(
                    (f"tiled_{tiles}x{tiles}of{pe}", gemm_tiled(tiles, pe, depth),
                     cycles_for(side, side, depth))
                )
    elif family == "attention":
        rows = cols = 16
        for groups in (1, 2, 4, 8, 16):
            if cols % groups:
                continue
            # One pass covers groups*rows of the reduction, so the pass count
            # falls as G rises; the array itself is the same size throughout.
            passes = -(-PROBLEM // (groups * rows))
            out.append(
                (f"attn_G{groups}", attention_pv(rows, cols, groups, 64),
                 passes * (64 + rows + cols))
            )
    else:
        raise SystemExit(f"unknown family {family!r}")
    return out


# The per-role report's summary table ends in a Total row whose columns are
# BRAM_18K, DSP, FF, LUT, URAM. Parsing anything else silently yields zeros,
# which is worse than failing: the sweep would report an area span of 0.0x and
# look like a finished measurement.
TOTAL_RE = re.compile(
    r"^\|Total\s*\|\s*(\S+)\|\s*(\S+)\|\s*(\S+)\|\s*(\S+)\|", re.M
)
FIELDS = ("BRAM", "DSP", "FF", "LUT")


def _number(token):
    return 0 if token.strip() in ("-", "") else int(token)


def role_area(out, name):
    """LUT/FF/DSP/BRAM for one role. Raises if the report cannot be parsed."""
    directory = os.path.join(out, name, "prj", "sol", "syn", "report")
    if not os.path.isdir(directory):
        raise RuntimeError(f"no HLS report directory for role {name}: {directory}")
    preferred = os.path.join(directory, f"{name}_0_csynth.rpt")
    if os.path.exists(preferred):
        path = preferred
    else:
        hits = sorted(
            f for f in os.listdir(directory)
            if f.endswith("_csynth.rpt") and "Pipeline" not in f
        )
        if not hits:
            raise RuntimeError(f"no csynth report for role {name} in {directory}")
        path = os.path.join(directory, hits[0])
    text = open(path, encoding="utf-8").read()
    start = text.find("== Utilization Estimates")
    if start < 0:
        raise RuntimeError(f"no utilization section in {path}")
    match = TOTAL_RE.search(text[start:])
    if match is None:
        raise RuntimeError(f"no Total row in the utilization table of {path}")
    return dict(zip(FIELDS, (_number(g) for g in match.groups())))


def sites_per_role(graph):
    emitter = UnitEmitter(graph)
    counts = {}
    for placement in emitter.placements():
        for order in range(len(emitter.classes(placement))):
            name = emitter.role_name(placement, order)
            _sig, _routing, sites = emitter.classes(placement)[order]
            counts[name] = len(sites)
    return counts


def cycles_for(rows, cols, depth):
    """Cycles to run a fixed PROBLEM^3 GEMM on a rows x cols array."""
    tiles = (
        -(-PROBLEM // rows) * -(-PROBLEM // cols) * -(-PROBLEM // depth)
    )
    per_tile = depth + rows + cols  # steady state plus fill and drain
    return tiles * per_tile


def pareto(points):
    """Indices of points not dominated on (cycles, LUT); both smaller is better."""
    keep = []
    for i, a in enumerate(points):
        dominated = any(
            j != i
            and b["cycles"] <= a["cycles"]
            and b["LUT"] <= a["LUT"]
            and (b["cycles"] < a["cycles"] or b["LUT"] < a["LUT"])
            for j, b in enumerate(points)
        )
        if not dominated:
            keep.append(i)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--part", default="xcu280-fsvh2892-2L-e")
    ap.add_argument("--frequency", type=float, default=300.0)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--shapes", default="4,8,16,32")
    ap.add_argument("--family", default="mesh", choices=("mesh", "tiled", "attention"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    root = args.out or f"/scratch/{os.environ.get('USER','x')}/table3"
    os.makedirs(root, exist_ok=True)
    dims = [int(x) for x in args.shapes.split(",")]

    points = []
    for tag, fabric, cycles in points_for(args.family, dims, args.depth):
        if True:
            out = os.path.join(root, tag)
            os.makedirs(out, exist_ok=True)
            start = time.time()
            graph = spmw.elaborate(fabric)
            names = stage(graph, out, args.part, args.frequency)
            elaborate_s = round(time.time() - start, 2)
            counts = sites_per_role(graph)
            synthesise(out, names, jobs=args.jobs)
            total = {"BRAM": 0, "DSP": 0, "FF": 0, "LUT": 0}
            missing = []
            for name in names:
                area = role_area(out, name)  # raises rather than returning zeros
                sites = counts.get(name, 0)
                if sites == 0:
                    missing.append(name)
                for key in total:
                    total[key] += area[key] * sites
            point = {
                "tag": tag,
                "roles": len(names),
                "instances": sum(counts.values()),
                "elaborate_s": elaborate_s,
                "cycles": cycles,
                "missing_reports": missing,
                **total,
            }
            points.append(point)
            print(
                "  {tag:12s} roles={roles:2d} inst={instances:5d} "
                "LUT={LUT:8d} DSP={DSP:6d} cycles={cycles:9d} "
                "elab={elaborate_s:5.2f}s".format(**point),
                flush=True,
            )

    front = pareto(points)
    luts = [p["LUT"] for p in points if p["LUT"] > 0]
    span = (max(luts) / min(luts)) if luts else 0.0

    print(f"\n=== Table 3: {args.family} ===")
    print(f"  structural knob : {args.family}")
    print(f"  points          : {len(points)}")
    print(f"  area span (LUT) : {span:.1f}x")
    print(f"  on the frontier : {len(front)}")
    print("\n  frontier points:")
    for i in sorted(front, key=lambda k: points[k]["cycles"]):
        p = points[i]
        print(f"    {p['tag']:12s} cycles={p['cycles']:9d} LUT={p['LUT']:8d}")

    broken = [p["tag"] for p in points if p["missing_reports"]]
    if broken:
        print(f"\nWARNING: missing HLS reports for {broken}; area is understated.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"points": points, "frontier": front, "span": span}, handle, indent=2)
        print(f"\nwrote {args.json}")
    print("TABLE3 DONE")


if __name__ == "__main__":
    main()
