# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tables of a controlled remeasurement (e9_simtime_controlled.py).

    python3 e9_simtime_tables.py <records.ndjson>

Only accepted runs count; the rejected ones are summed at the end with why.
Each cell is the median over every accepted timed run, with the range after
it. A hot run is any run of an already compiled simulation; a cold one is the
build and the first run together.
"""

import collections
import json
import statistics
import sys

PROGRAMS = ("llama", "dsv4")
SIZES = (4, 8, 16)
NAMES = {"llama": "LLaMA", "dsv4": "DeepSeek"}
UNITS = {4: 42, 8: 114, 16: 354}  # one thread a unit: S^2 + 6S + 2


def cell(values, digits=2):
    if not values:
        return "-"
    med = statistics.median(values)
    lo, hi = min(values), max(values)
    fmt = f"{{:.{digits}f}}"
    if fmt.format(lo) == fmt.format(hi):
        return fmt.format(med)
    return f"{fmt.format(med)} ({fmt.format(lo)}-{fmt.format(hi)})"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def main():
    records = [
        json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()
    ]
    ok = [r for r in records if r.get("accepted")]
    by = collections.defaultdict(list)
    for r in ok:
        by[(r["phase"], r["program"], r["size"], r.get("cpus"))].append(r)

    for phase in ("sim64", "sim0"):
        rows = []
        for program in PROGRAMS:
            for size in SIZES:
                rs = by.get((phase, program, size, None), [])
                if not rs:
                    continue
                hot = [h for r in rs for h in r["runs"][1:]]
                rows.append(
                    [
                        NAMES[program],
                        f"{size}x{size}",
                        UNITS[size],
                        len(rs),
                        cell([r["build_s"] for r in rs], 1),
                        cell([r["build_s"] + r["runs"][0]["s"] for r in rs], 1),
                        cell([h["s"] for h in hot], 3),
                        cell([h["cores"] for h in hot], 1),
                        cell([h["nvcsw"] / h["s"] / 1000 for h in hot], 0),
                        cell([h["nivcsw"] / h["s"] / 1000 for h in hot], 0),
                        cell([h["others_cores"] for r in rs for h in r["runs"]], 2),
                    ]
                )
        if rows:
            print(f"\n## {phase}: the OpenMP simulator, a thread a unit\n")
            print(
                table(
                    [
                        "Program",
                        "Engine",
                        "Threads",
                        "Processes",
                        "Build s",
                        "Cold s",
                        "Hot s",
                        "Cores busy",
                        "Voluntary switches k/s",
                        "Involuntary k/s",
                        "Others' cores",
                    ],
                    rows,
                )
            )

    rows = []
    for size in SIZES:
        row = [f"{size}x{size}", UNITS[size]]
        for cpus in (12, 24, 48):
            rs = by.get(("cores", "llama", size, cpus), [])
            row.append(cell([h["s"] for r in rs for h in r["runs"][1:]], 3))
        rows.append(row)
    if any(c != "-" for r in rows for c in r[2:]):
        print("\n## cores: LLaMA hot, floor 64, pinned to a CPU set\n")
        print(table(["Engine", "Threads", "12 CPUs", "24 CPUs", "48 CPUs"], rows))

    rows = []
    for program in PROGRAMS:
        for size in SIZES:
            rs = by.get(("csim", program, size, None), [])
            if not rs:
                continue
            rows.append(
                [
                    NAMES[program],
                    f"{size}x{size}",
                    len(rs),
                    cell([r["codegen_s"] + r["csim_design_s"] for r in rs], 1),
                    cell([h for r in rs for h in r["hot_s"]], 2),
                    cell(
                        [
                            x["others_cores"]
                            for r in rs
                            for x in [r["vitis_run"], *r["hot_runs"]]
                        ],
                        2,
                    ),
                ]
            )
    if rows:
        print("\n## csim: Vitis HLS's C simulation\n")
        print(
            table(
                ["Program", "Engine", "Runs", "Cold s", "Hot s", "Others' cores"], rows
            )
        )

    rows = []
    for program in PROGRAMS:
        for size in SIZES:
            rs = by.get(("cosim", program, size, None), [])
            if not rs:
                continue
            steps = [r["cosim_steps_s"] or {} for r in rs]
            rows.append(
                [
                    NAMES[program],
                    f"{size}x{size}",
                    len(rs),
                    cell([s.get("xsim") for s in steps if s.get("xsim")], 1),
                    cell([r["hls_wall_s"] for r in rs if r.get("hls_wall_s")], 1),
                    cell([r["elaborate_s"] for r in rs if r.get("elaborate_s")], 1),
                    cell([r["wall_s"] for r in rs], 0),
                    cell(
                        [
                            r["xsim_others_mean"]
                            for r in rs
                            if r.get("xsim_others_mean") is not None
                        ],
                        2,
                    ),
                    sorted({int(r["cycles"]) for r in rs if r.get("cycles")}),
                ]
            )
    if rows:
        print("\n## cosim: Vivado RTL\n")
        print(
            table(
                [
                    "Program",
                    "Engine",
                    "Runs",
                    "xsim s",
                    "HLS s",
                    "Elaborate s",
                    "End to end s",
                    "Others' cores in xsim",
                    "Cycles",
                ],
                rows,
            )
        )

    rejected = [r for r in records if not r.get("accepted")]
    if rejected:
        print(f"\n{len(rejected)} attempts rejected:")
        for r in rejected:
            print(
                f"  {r['phase']} {r['program']} {r['size']}x{r['size']} attempt {r['attempt']}: {r['rejected_for']}"
            )
    waited = sum(r.get("gate_wait_s", 0) for r in records)
    span = (
        max(r["t_end"] for r in records)
        - min(r["t_start"] - r.get("gate_wait_s", 0) for r in records)
    ) / 3600
    print(
        f"\n{len(ok)} accepted runs over {span:.1f} h, {waited / 3600:.1f} h of it waiting for a quiet machine"
    )


if __name__ == "__main__":
    main()
