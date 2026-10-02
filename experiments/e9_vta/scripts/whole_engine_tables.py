# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tables of "Whole engines, run by their own instructions".

    python3 whole_engine_tables.py [experiments/e9_vta]

Everything is read from the files beside this script:

* `gemmini_full/routes.json` -- every route of Gemmini's whole accelerator and
  of VTA's `Core`, with Gemmini's split by block and both engines' scope
  timings (`collect_routes.py`);
* `gemmini_full/sim/results.txt`, `vta_core/results.txt`,
  `ptpu_mem/results.txt` -- the result line of every instruction-level
  simulation, of Gemmini, VTA and SPMW's whole engine;
* `ptpu_mem/routes.txt` -- every route of SPMW's whole engine, and
  `ptpu_mem/S<n>/report/[p30_]area_by_unit.txt` its split by unit;
* `results.csv` -- the cycles of SPMW's programmable array, fed by streams,
  and `ptpu/routes.txt` its routes;
* `core/split.txt` -- VTA's datapath + scratchpads.

Per design the route with the best clock is the one reported. Gemmini's end
to end is the faster of the library's and the hand-scheduled program on the
64-bit bus, and its execute only the fewest cycles its execute controller is
busy in any run of that workload, on either bus: for the microbenchmark that
is the run with every load issued first (`microp`).
"""

import csv
import json
import os
import re
import sys

SIZES = (4, 8, 16)
WORKLOADS = (
    ("micro", "E3 micro, 16 tiles", 16 * 16**3),
    ("llama", "LLaMA slice", 64 * 2048 * 128),
    ("dsv4", "DeepSeek slice", 64 * 7168 * 128),
)
CONFIGS = (
    ("matmul", "Gemmini, matmul only"),
    ("shift", "Gemmini, integer shift"),
    ("lean", "Gemmini, as shipped"),
)
BLOCKS = (
    ("execute controller + mesh", "execute controller and mesh"),
    ("of which mesh", "&nbsp;&nbsp;of which the mesh"),
    ("scratchpad, accumulator, scale", "scratchpad, accumulator, scale"),
    ("DMA", "DMA"),
    ("TLB", "TLB"),
    ("load controller", "load controller"),
    ("store controller", "store controller"),
    ("command queues", "command queues"),
    ("reservation station", "reservation station"),
    ("loop unrollers", "loop unrollers"),
    ("counters", "counters"),
)
CSV_NAMES = {
    "micro": "micro-fixed16",
    "llama": "llama-gateup-slice",
    "dsv4": "dsv4-gateup-slice",
}
TARGET = 3.333
#: SPMW's whole engine by unit, as `area_by_unit.txt` names them: what decodes
#: and executes with the storage it works from, and what moves data to it and
#: from it.
EXECUTE = (
    "cells",
    "links between cells",
    "edge taps and their links",
    "head",
    "micro-op queue, taps and links",
    "lanes, with their accumulators",
    "result taps and their links",
)
MEMORY = (
    "requester",
    "dealer",
    "operand queues to the head",
    "row buffer and its credits",
    "packer",
    "write requester",
)


def results(path):
    """``{run: {field: int}}`` from a file of bench result lines."""
    runs = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            name, _, rest = line.partition(" ")
            if " RESULT " in line:
                runs[name] = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", rest)}
    return runs


def load(root):
    """Every number the tables need."""
    with open(os.path.join(root, "gemmini_full/routes.json"), encoding="utf-8") as f:
        routes = json.load(f)
    spmw = {S: {} for S in SIZES}
    with open(os.path.join(root, "results.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["engine"] != "spmw-programmable":
                continue
            S = int(row["S"])
            total = re.search(r"total (\d+)", row["note"])
            spmw[S][row["workload"]] = (
                int(total.group(1)) if total else int(row["cycles_per_tile"])
            )
            spmw[S].update(
                lut=int(row["lut"]),
                ff=int(row["ff"]),
                period=TARGET - float(row["wns_ns"]),
            )
    scope = {}
    with open(os.path.join(root, "core/split.txt"), encoding="utf-8") as f:
        for line in f:
            hit = re.match(
                r"w(\d+): .*datapath\+scratchpads (\d+) LUT (\d+) FF (\d+) BRAM (\d+) URAM",
                line,
            )
            if hit:
                scope[int(hit.group(1))] = tuple(int(g) for g in hit.groups()[1:])
    runs = whole_routes(os.path.join(root, "ptpu_mem/routes.txt"))
    fed = whole_routes(os.path.join(root, "ptpu/routes.txt"))
    for S in SIZES:
        spmw[S].update(
            lut=fed[S][0]["lut"], ff=fed[S][0]["ff"], period=fed[S][0]["period"]
        )
    return {
        "routes": routes,
        "gem": results(os.path.join(root, "gemmini_full/sim/results.txt")),
        "vta": results(os.path.join(root, "vta_core/results.txt")),
        "mem": results(os.path.join(root, "ptpu_mem/results.txt")),
        "spmw": spmw,
        "whole_routes": runs,
        "fed_routes": fed,
        "whole": {S: runs[S][0] for S in SIZES},
        "units": {S: unit_split(root, S, runs[S][0]) for S in SIZES},
        "vta_scope": scope,
    }


def whole_routes(path):
    """One SPMW engine's routes by size, the best clock first."""
    runs = {S: [] for S in SIZES}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ROUTE "):
                continue
            f = dict(re.findall(r"(\w+)=(\S+)", line))
            runs[int(f["size"])].append(
                {
                    "target": float(f["target"]),
                    "period": float(f["period"]),
                    "lut": int(f["lut"]),
                    "ff": int(f["ff"]),
                    "bram": int(f.get("ramb36", 0)) + int(f.get("ramb18", 0)) / 2,
                    "dir": f["dir"],
                }
            )
    return {
        S: sorted(group, key=lambda run: run["period"]) for S, group in runs.items()
    }


def unit_split(root, S, run):
    """``{unit: (LUT, FF, RAMB36, RAMB18)}`` of one route of the whole engine."""
    prefix = "" if run["dir"].startswith("b_") else run["dir"].split("_")[-1] + "_"
    path = os.path.join(root, f"ptpu_mem/S{S}/report/{prefix}area_by_unit.txt")
    split = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            hit = re.match(
                r"  (\S.*?) +n= *\d+ LUT= *(\d+) FF= *(\d+) RAMB36= *(\d+) RAMB18= *(\d+)",
                line,
            )
            if hit:
                split[hit.group(1)] = tuple(int(g) for g in hit.groups()[1:])
    return split


def part(data, S, names):
    """The lookup tables, registers and block RAM tiles of some of the units."""
    rows = [data["units"][S][name] for name in names]
    return (
        sum(r[0] for r in rows),
        sum(r[1] for r in rows),
        sum(r[2] + r[3] / 2 for r in rows),
    )


def best(entry):
    """The best clock of the routes of one design."""
    return min(run["period"] for run in entry["runs"])


def scope_run(entry):
    """The route on which a design's scope has its best clock."""
    timed = [run for run in entry["runs"] if "scope_period" in run]
    return min(timed, key=lambda run: run["scope_period"])


def scope_best(entry):
    """The best clock of a design's scope, over the routes it was timed on."""
    return scope_run(entry)["scope_period"]


def run_of(entry):
    """The route whose area is reported: the one with the best clock."""
    return next(run for run in entry["runs"] if run["dir"] == entry["best"])


def gemmini(data, config, workload, S, bus=""):
    """Gemmini's faster program for a workload: its result fields."""
    names = [f"{workload}_{config}_{S}{bus}"]
    if workload == "micro":
        names.append(f"microb_{config}_{S}{bus}")
    runs = [data["gem"][name] for name in names if name in data["gem"]]
    return min(runs, key=lambda run: run["total"]) if runs else None


def execute(data, workload, S):
    """The fewest cycles Gemmini's execute controller is busy, in any run."""
    tags = ("micro", "microb", "microp") if workload == "micro" else (workload,)
    names = [f"{tag}_matmul_{S}{bus}" for tag in tags for bus in ("", "_b128")]
    return min(data["gem"][name]["ex"] for name in names if name in data["gem"])


def mhz(period):
    return f"{1000 / period:.0f} MHz"


def clock(seconds, workload):
    """A time to three figures, in the unit its workload is read in."""
    scale, unit = (1e6, "us") if workload == "micro" else (1e3, "ms")
    return f"{seconds * scale:#.3g}".rstrip(".") + " " + unit


def hardware(data):
    """Each engine, whole."""
    routes = data["routes"]
    print("| array | engine | LUT | FF | BRAM | URAM | DSP | clock |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for S in SIZES:
        s = data["whole"][S]
        print(
            f"| {S}x{S} | SPMW | {s['lut']:,} | {s['ff']:,} | {s['bram']:g} "
            f"| 0 | 0 | {mhz(s['period'])} |"
        )
        for config, label in CONFIGS:
            g = routes["gem"].get(config, {}).get(str(S))
            if g:
                print(
                    f"| | {label} | {g['lut']:,} | {g['ff']:,} | {g['bram']:g} "
                    f"| {g['uram']} | {g['dsp']} | {mhz(best(g))} |"
                )
        v = routes["vta"][str(S)]
        print(
            f"| | VTA | {v['lut']:,} | {v['ff']:,} | {v['bram']:g} | {v['uram']} "
            f"| {v['dsp']} | {mhz(best(v))} |"
        )


def scopes(data):
    """The execute scope of each engine."""
    routes = data["routes"]
    print("| array | engine, execute scope | LUT | FF | BRAM | URAM | clock |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for S in SIZES:
        s = data["spmw"][S]
        print(
            f"| {S}x{S} | SPMW, the array and its dispatch, fed by streams "
            f"| {s['lut']:,} | {s['ff']:,} | 0 | 0 | {mhz(s['period'])} |"
        )
        lut, ff, bram = part(data, S, EXECUTE)
        print(
            f"| | SPMW, the whole engine less its read and write sides "
            f"| {lut:,} | {ff:,} | {bram:g} | 0 | {mhz(data['whole'][S]['period'])} |"
        )
        g = routes["gem"]["matmul"].get(str(S))
        if g:
            lut, ff, bram, uram, _ = scope_run(g)["scope"]
            print(
                f"| | Gemmini, execute controller, mesh and storage | {lut:,} "
                f"| {ff:,} | {bram:g} | {uram} | {mhz(scope_best(g))} |"
            )
        lut, ff, bram, uram = data["vta_scope"][S]
        print(
            f"| | VTA, datapath and scratchpads | {lut:,} | {ff:,} | {bram} "
            f"| {uram} | {mhz(scope_best(routes['vta'][str(S)]))} |"
        )


def units(data):
    """SPMW's whole engine by unit, at the smallest size and the largest."""
    print(
        "| unit | 4x4 LUT | FF | 16x16 LUT | FF | block RAM tiles, 4x4 / 8x8 / 16x16 |"
    )
    print("|---|---:|---:|---:|---:|---|")
    for name in EXECUTE + MEMORY:
        small, large = data["units"][4][name], data["units"][16][name]
        tiles = [
            data["units"][S][name][2] + data["units"][S][name][3] / 2 for S in SIZES
        ]
        ram = " / ".join(f"{t:g}" for t in tiles) if any(tiles) else ""
        print(
            f"| {name} | {small[0]:,} | {small[1]:,} | {large[0]:,} | {large[1]:,} "
            f"| {ram} |"
        )
    for label, names in (("the read and write sides", MEMORY), ("the rest", EXECUTE)):
        cells = ", ".join(
            f"{S}x{S} {part(data, S, names)[0]:,} LUT {part(data, S, names)[1]:,} FF "
            f"{part(data, S, names)[2]:g} BRAM"
            for S in SIZES
        )
        print(f"\n{label}: {cells}")
    print("\nagainst the stream-fed array, both routed at 3.333 ns:\n")
    at = lambda runs: next(run for run in runs if run["target"] == TARGET)
    print("| array | | stream-fed | whole engine | |")
    print("|---|---|---:|---:|---:|")
    for S in SIZES:
        w, a = at(data["whole_routes"][S]), at(data["fed_routes"][S])
        print(
            f"| {S}x{S} | LUT | {a['lut']:,} | {w['lut']:,} "
            f"| {w['lut'] / a['lut'] - 1:+.0%} |"
        )
        print(f"| | FF | {a['ff']:,} | {w['ff']:,} | {w['ff'] / a['ff'] - 1:+.0%} |")
        print(f"| | block RAM | 0 | {w['bram']:g} | |")
        print(
            f"| | clock | {mhz(a['period'])} | {mhz(w['period'])} "
            f"| {a['period'] / w['period'] - 1:+.1%} |"
        )


def blocks(data):
    """The matmul-only Gemmini by block."""
    matmul = data["routes"]["gem"]["matmul"]
    have = [S for S in SIZES if str(S) in matmul]
    print("| block | " + " | ".join(f"{S}x{S} LUT | FF" for S in have) + " |")
    print("|---|" + "---:|---:|" * len(have))
    for name, label in BLOCKS:
        cells = []
        for S in have:
            lut, ff = run_of(matmul[str(S)])["blocks"].get(name, (0, 0))[:2]
            cells.append(f"{lut:,} | {ff:,}")
        print(f"| {label} | " + " | ".join(cells) + " |")
    cells = [
        f"**{matmul[str(S)]['lut']:,}** | **{matmul[str(S)]['ff']:,}**" for S in have
    ]
    print("| whole accelerator | " + " | ".join(cells) + " |")


def cycles(data):
    """Cycles, end to end and in the execute scope."""
    print(
        "| workload | array | floor | SPMW | Gemmini | VTA "
        "| SPMW, array only | Gemmini, execute only | VTA, compute only |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for workload, label, macs in WORKLOADS:
        for S in SIZES:
            g = gemmini(data, "matmul", workload, S)
            v = data["vta"][f"{workload}_w{S}"]
            print(
                f"| {label if S == SIZES[0] else ''} | {S}x{S} | {macs // S**2:,} "
                f"| **{data['mem'][f'{workload}_{S}']['total']:,}** | {g['total']:,} "
                f"| {v['total']:,} | {data['spmw'][S][CSV_NAMES[workload]]:,} "
                f"| {execute(data, workload, S):,} | {v['gemm'] + v['alu']:,} |"
            )


def whole_time(data, workload, S):
    """SPMW's whole engine end to end, in seconds at its routed clock."""
    return data["mem"][f"{workload}_{S}"]["total"] * data["whole"][S]["period"] * 1e-9


def times(data):
    """Time end to end, at each whole engine's routed clock."""
    routes = data["routes"]
    print(
        "| workload | array | SPMW | Gemmini, matmul only "
        "| Gemmini, as shipped | VTA | Gemmini / SPMW | VTA / SPMW |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for workload, label, _ in WORKLOADS:
        for S in SIZES:
            ts = whole_time(data, workload, S)
            cells, tg = [], None
            for config in ("matmul", "lean"):
                g = routes["gem"].get(config, {}).get(str(S))
                run = gemmini(data, config, workload, S)
                t = run["total"] * best(g) * 1e-9 if g and run else None
                cells.append(clock(t, workload) if t else "...")
                tg = tg or t
            tv = (
                data["vta"][f"{workload}_w{S}"]["total"]
                * best(routes["vta"][str(S)])
                * 1e-9
            )
            over = lambda t, base: f"{t / base:.2f}x" if t else "..."
            print(
                f"| {label if S == SIZES[0] else ''} | {S}x{S} "
                f"| **{clock(ts, workload)}** | {cells[0]} | {cells[1]} "
                f"| {clock(tv, workload)} | {over(tg, ts)} | {over(tv, ts)} |"
            )


def stalls(data):
    """SPMW's whole engine on a memory that withholds a quarter of its beats."""
    print("| workload | array | floor | ideal memory | stalling memory | change |")
    print("|---|---|---:|---:|---:|---:|")
    for workload, label, macs in WORKLOADS[:2] + (("mixed", "five GEMMs", 0),):
        for S in SIZES:
            ideal = data["mem"][f"{workload}_{S}"]["total"]
            slow = data["mem"][f"{workload}_{S}_stall25"]["total"]
            floor = f"{macs // S**2:,}" if macs else ""
            print(
                f"| {label if S == SIZES[0] else ''} | {S}x{S} | {floor} "
                f"| {ideal:,} | {slow:,} | {slow / ideal - 1:+.1%} |"
            )


def scope_times(data):
    """Time in the execute scope, and lookup tables times time."""
    routes = data["routes"]
    print(
        "| workload | array | SPMW | Gemmini | VTA | Gemmini / SPMW | VTA / SPMW "
        "| Gemmini, LUT x time | VTA, LUT x time |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for workload, label, _ in WORKLOADS:
        for S in SIZES:
            s = data["spmw"][S]
            ts = s[CSV_NAMES[workload]] * s["period"] * 1e-9
            v = data["vta"][f"{workload}_w{S}"]
            tv = (v["gemm"] + v["alu"]) * scope_best(routes["vta"][str(S)]) * 1e-9
            area = ts * s["lut"]
            g = routes["gem"]["matmul"].get(str(S))
            cells = ("...", "...", "...")
            if g and any("scope_period" in run for run in g["runs"]):
                tg = execute(data, workload, S) * scope_best(g) * 1e-9
                cells = (
                    clock(tg, workload),
                    f"{tg / ts:.2f}x",
                    f"{tg * scope_run(g)['scope'][0] / area:.1f}x",
                )
            print(
                f"| {label if S == SIZES[0] else ''} | {S}x{S} | **{clock(ts, workload)}** "
                f"| {cells[0]} | {clock(tv, workload)} | {cells[1]} "
                f"| {tv / ts:.2f}x | {cells[2]} "
                f"| {tv * data['vta_scope'][S][0] / area:.1f}x |"
            )


def bus(data):
    """Gemmini on Chipyard's 128-bit system bus against the 64-bit one."""
    print("| workload | array | 64-bit bus | 128-bit bus | change |")
    print("|---|---|---:|---:|---:|")
    for workload, label, _ in WORKLOADS:
        for S in SIZES:
            narrow = gemmini(data, "matmul", workload, S)
            wide = gemmini(data, "matmul", workload, S, "_b128")
            if wide:
                change = wide["total"] / narrow["total"] - 1
                moved = f"{change:+.1%}" if abs(change) >= 0.0005 else "0.0%"
                print(
                    f"| {label} | {S}x{S} | {narrow['total']:,} "
                    f"| {wide['total']:,} | {moved} |"
                )
                label = ""


def detail(data):
    """What the prose quotes: ratios to the floor and every route's clock."""
    print(
        "over the floor: SPMW, Gemmini, VTA end to end | SPMW's array, Gemmini, "
        "VTA in scope | end-to-end time over SPMW's: Gemmini, VTA "
        "| end-to-end LUT x time over SPMW's: Gemmini, VTA"
    )
    routes = data["routes"]
    for workload, _, macs in WORKLOADS:
        for S in SIZES:
            floor = macs // S**2
            g = gemmini(data, "matmul", workload, S)
            v = data["vta"][f"{workload}_w{S}"]
            total = data["mem"][f"{workload}_{S}"]["total"]
            ts = whole_time(data, workload, S) * 1e9
            gem, vta = routes["gem"]["matmul"][str(S)], routes["vta"][str(S)]
            tg = g["total"] * best(gem)
            tv = v["total"] * best(vta)
            area = ts * data["whole"][S]["lut"]
            print(
                f"  {workload:6s} {S:2d}: {total / floor:.4f} "
                f"{g['total'] / floor:.3f} {v['total'] / floor:.3f} "
                f"| {data['spmw'][S][CSV_NAMES[workload]] / floor:.3f} "
                f"{execute(data, workload, S) / floor:.3f} "
                f"{(v['gemm'] + v['alu']) / floor:.3f} "
                f"| {tg / ts:.2f} {tv / ts:.2f} "
                f"| {tg * gem['lut'] / area:.1f} {tv * vta['lut'] / area:.1f}"
            )
    print("every route, target -> period in ns (scope period):")
    for name, group in (("whole", "whole_routes"), ("stream-fed", "fed_routes")):
        for S in SIZES:
            runs = ", ".join(f"{r['target']} -> {r['period']}" for r in data[group][S])
            print(f"  spmw {name} {S}: {runs}")
    designs = [(f"gemmini {c}", data["routes"]["gem"].get(c, {})) for c, _ in CONFIGS]
    for name, group in designs + [("vta", data["routes"]["vta"])]:
        for S in SIZES:
            entry = group.get(str(S))
            if entry:
                runs = ", ".join(
                    f"{r['target']} -> {r['period']}"
                    + (f" ({r['scope_period']})" if "scope_period" in r else "")
                    for r in entry["runs"]
                )
                print(f"  {name} {S}: {runs}")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = load(sys.argv[1] if len(sys.argv) > 1 else here)
    for title, table in (
        ("The hardware, whole", hardware),
        ("The hardware, execute scope", scopes),
        ("SPMW by unit", units),
        ("Gemmini by block", blocks),
        ("Cycles", cycles),
        ("Time, end to end", times),
        ("SPMW on a stalling memory", stalls),
        ("Time, execute scope", scope_times),
        ("Gemmini's memory bus", bus),
        ("Detail", detail),
    ):
        print(f"## {title}\n")
        table(data)
        print()


if __name__ == "__main__":
    main()
