# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""split.py <pnr dir>...: a routed Gemmini by its top-level blocks.

The scratchpad block holds both Gemmini's storage and its DMA, so it is split
one level further: the stream reader and writer and their TileLink plumbing
are the DMA, and the rest is the scratchpad and accumulator memories and the
scale units. `execute scope` is then the execute controller, the mesh and
that storage: the part with a counterpart in SPMW's programmable engine and
in VTA's datapath and scratchpads.
"""
import re
import sys

NAMES = {
    "ex_controller": "execute controller + mesh",
    "load_controller": "load controller",
    "store_controller": "store controller",
    "reservation_station": "reservation station",
    "reservation_station_completed_arb": "reservation station",
    "mod": "loop unrollers",
    "mod_1": "loop unrollers",
    "tlb": "TLB",
    "counters": "counters",
    "im2col": "im2col",
    "raw_cmd_q": "command queues",
    "unrolled_cmd_q": "command queues",
}
DMA = ("reader", "writer", "buffer", "xbar", "widget")
SCOPE = ("execute controller + mesh", "scratchpad, accumulator, scale")
ZERO = (0, 0, 0, 0, 0)


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def blocks(d):
    """``(whole, {block: (LUT, FF, BRAM, URAM, DSP)})`` of one routed design."""
    rows, cols, top, inside = {}, None, ZERO, False
    for line in open(d + "/util_hier.rpt", errors="replace"):
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if not cells:
            continue
        if cells[0] == "Instance":
            cols = cells
            continue
        if cols is None or len(cells) != len(cols):
            continue
        raw = line.split("|")[1]
        depth = (len(raw) - len(raw.lstrip())) // 2
        get = lambda k: int(cells[cols.index(k)])
        vals = (
            get("Total LUTs"),
            get("FFs"),
            get("RAMB36") + get("RAMB18") / 2,
            get("URAM"),
            get("DSP Blocks"),
        )
        name = cells[0]
        if depth == 0:
            top = vals
        elif depth == 1 and not name.startswith("("):
            inside = name == "spad"
            if not inside:
                key = NAMES.get(name, name)
                rows[key] = add(rows.get(key, ZERO), vals)
        elif depth == 2 and inside:
            key = "DMA" if name.startswith(DMA) else "scratchpad, accumulator, scale"
            rows[key] = add(rows.get(key, ZERO), vals)
        elif depth == 2 and name == "mesh":
            rows["  of which mesh"] = vals
    return top, rows


def scope(rows):
    """The execute scope of a split."""
    total = ZERO
    for key in SCOPE:
        total = add(total, rows.get(key, ZERO))
    return total


if __name__ == "__main__":
    for d in sys.argv[1:]:
        top, rows = blocks(d)
        wns = float(
            re.search(
                r"WNS\(ns\).*?\n.*?\n\s*(-?[\d.]+)",
                open(d + "/timing.rpt").read(),
                re.S,
            ).group(1)
        )
        period = float(
            re.search(
                r"create_clock -period ([\d.]+)", open(d + "/pnr.tcl").read()
            ).group(1)
        )
        line = "%-38s LUT %6d FF %6d BRAM %5g URAM %2d DSP %3d"
        print(
            f"== {d.rstrip('/').split('/')[-1]}: target {period} ns, WNS {wns:+.3f}, {1000 / (period - wns):.1f} MHz"
        )
        print("  " + line % (("whole accelerator",) + top))
        print("  " + line % (("execute scope",) + scope(rows)))
        for k, v in sorted(rows.items(), key=lambda kv: -kv[1][0]):
            print("    " + line % ((k,) + v))
