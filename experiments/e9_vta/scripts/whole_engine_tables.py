# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tables of "Whole engines, run by their own instructions".

    python3 whole_engine_tables.py [experiments/e9_vta]

Everything is read from the files beside this script:

* `gemmini_full/routes.json` -- every route of Gemmini's whole accelerator and
  of VTA's `Core`, with Gemmini's split by block and both engines' scope
  timings (`collect_routes.py`);
* `gemmini_full/sim/results.txt`, `vta_core/results.txt` -- the result line
  of every instruction-level simulation;
* `results.csv` -- SPMW's programmable engine;
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
    return {
        "routes": routes,
        "gem": results(os.path.join(root, "gemmini_full/sim/results.txt")),
        "vta": results(os.path.join(root, "vta_core/results.txt")),
        "spmw": spmw,
        "vta_scope": scope,
    }


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
        s = data["spmw"][S]
        print(
            f"| {S}x{S} | SPMW, no memory system | {s['lut']:,} | {s['ff']:,} "
            f"| 0 | 0 | 0 | {mhz(s['period'])} |"
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
            f"| {S}x{S} | SPMW, the whole programmable engine | {s['lut']:,} "
            f"| {s['ff']:,} | 0 | 0 | {mhz(s['period'])} |"
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
        "| Gemmini, execute only | VTA, compute only |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for workload, label, macs in WORKLOADS:
        for S in SIZES:
            g = gemmini(data, "matmul", workload, S)
            v = data["vta"][f"{workload}_w{S}"]
            print(
                f"| {label if S == SIZES[0] else ''} | {S}x{S} | {macs // S**2:,} "
                f"| **{data['spmw'][S][CSV_NAMES[workload]]:,}** | {g['total']:,} "
                f"| {v['total']:,} | {execute(data, workload, S):,} "
                f"| {v['gemm'] + v['alu']:,} |"
            )


def times(data):
    """Time end to end, at each engine's routed clock.

    SPMW's array has no memory system, so its column is a bound, and the
    ratios to it are left to `detail`.
    """
    routes = data["routes"]
    print(
        "| workload | array | SPMW, no memory system | Gemmini, matmul only "
        "| Gemmini, as shipped | VTA | Gemmini / VTA |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")
    for workload, label, _ in WORKLOADS:
        for S in SIZES:
            s = data["spmw"][S]
            ts = s[CSV_NAMES[workload]] * s["period"] * 1e-9
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
                f"| {label if S == SIZES[0] else ''} | {S}x{S} | {clock(ts, workload)} "
                f"| {cells[0]} | {cells[1]} | {clock(tv, workload)} | {over(tg, tv)} |"
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
        "over the floor: SPMW, Gemmini, VTA end to end | Gemmini, VTA in scope "
        "| end-to-end time over SPMW's: Gemmini, VTA"
    )
    routes = data["routes"]
    for workload, _, macs in WORKLOADS:
        for S in SIZES:
            floor = macs // S**2
            g = gemmini(data, "matmul", workload, S)
            v = data["vta"][f"{workload}_w{S}"]
            s = data["spmw"][S]
            ts = s[CSV_NAMES[workload]] * s["period"]
            tg = g["total"] * best(routes["gem"]["matmul"][str(S)])
            tv = v["total"] * best(routes["vta"][str(S)])
            print(
                f"  {workload:6s} {S:2d}: {data['spmw'][S][CSV_NAMES[workload]] / floor:.3f} "
                f"{g['total'] / floor:.3f} {v['total'] / floor:.3f} "
                f"| {execute(data, workload, S) / floor:.3f} "
                f"{(v['gemm'] + v['alu']) / floor:.3f} "
                f"| {tg / ts:.2f} {tv / ts:.2f}"
            )
    print("every route, target -> period in ns (scope period):")
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
        ("Gemmini by block", blocks),
        ("Cycles", cycles),
        ("Time, end to end", times),
        ("Time, execute scope", scope_times),
        ("Gemmini's memory bus", bus),
        ("Detail", detail),
    ):
        print(f"## {title}\n")
        table(data)
        print()


if __name__ == "__main__":
    main()
