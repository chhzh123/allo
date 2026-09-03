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
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    root = args.out or f"/scratch/{os.environ.get('USER','x')}/table3"
    os.makedirs(root, exist_ok=True)
    dims = [int(x) for x in args.shapes.split(",")]

    points = []
    for rows in dims:
        for cols in dims:
            tag = f"mesh_{rows}x{cols}"
            out = os.path.join(root, tag)
            os.makedirs(out, exist_ok=True)
            start = time.time()
            graph = spmw.elaborate(gemm_mesh(rows, cols, args.depth))
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
                "rows": rows,
                "cols": cols,
                "roles": len(names),
                "instances": sum(counts.values()),
                "elaborate_s": elaborate_s,
                "cycles": cycles_for(rows, cols, args.depth),
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

    print("\n=== Table 3: GEMM mesh ===")
    print(f"  structural knob : rows x columns")
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
