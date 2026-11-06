#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SYCL compile-time table, from report/results.csv and report/split/.

    python3 scripts/sycl_table.py

Medians over the repetitions, with the range when there is more than one.
The E1 columns are read from ../e1_compile_time.csv and the first table of
../README.md quotes the same numbers."""
import csv
import os
import re
import statistics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E5 = os.path.dirname(HERE)

runs = {}
for row in csv.DictReader(open(os.path.join(HERE, "report", "results.csv"))):
    if row["rc"] == "0":
        runs.setdefault(int(row["side"]), []).append(row)

host = {}
for line in open(os.path.join(HERE, "report", "split", "split_raw.txt")):
    side, rest = line.strip().split(",", 1)
    host[int(side)] = float(re.search(r"host=([\d.]+)", rest).group(1))

e1 = {}
for row in csv.DictReader(open(os.path.join(E5, "e1_compile_time.csv"))):
    side = int(row["array"].split("x")[0])
    key = (row["framework"], row["stage"])
    if key in (("autosa", "hls_csynth"), ("allo", "csynth")) and row["interface"] in (
        "narrow",
        "32b",
    ):
        e1[(side, row["framework"])] = float(row["seconds"])
    if key == ("spmw_mesh", "hls_csynth_export"):
        e1[(side, "spmw")] = float(row["seconds"])
    if key == ("spmw_mesh", "hls_sum_of_jobs"):
        e1[(side, "spmw_serial")] = float(row["seconds"])


def cell(values):
    mid = statistics.median(values)
    fmt = "{:,.1f}" if mid < 100 else "{:,.0f}"
    text = fmt.format(mid) + " s"
    if len(values) > 1:
        text += " (" + fmt.format(min(values)) + "-" + fmt.format(max(values)) + ")"
    return text


print(
    "| Array | Kernels (cells) | Intel SYCL, whole compile | Intel HLS backend | Kernels only (vs csynth) "
    "| AutoSA csynth | Allo csynth | SPMW, 8 workers | SPMW, one worker | Peak memory |"
)
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for side in sorted(runs):
    rows = runs[side]
    wall = [float(r["wall_s"]) for r in rows]
    hls = [float(r["hls_done_s"]) - float(r["aoc_start_s"]) for r in rows]
    kernel_only = [float(r["hls_done_s"]) - host[side] for r in rows]
    rss = max(int(r["max_rss_kb"]) for r in rows) / 1048576
    r0 = rows[0]
    print(
        "| %dx%d | %s (%s) | %s | %s | %s | %s | %s | %s | %s | %.1f GB |"
        % (
            side,
            side,
            "{:,}".format(int(r0["kernels"])),
            "{:,}".format(int(r0["cells"])),
            cell(wall),
            cell(hls),
            cell(kernel_only),
            "{:,.0f} s".format(e1[(side, "autosa")]),
            "{:,.0f} s".format(e1[(side, "allo")]) if (side, "allo") in e1 else "",
            "{:,.0f} s".format(e1[(side, "spmw")]),
            "{:,.0f} s".format(e1[(side, "spmw_serial")]),
            rss,
        )
    )
print()
print(
    "| Array | Repetitions | CPU / wall | Load at start | Device front end | Host compile | Link | To SPIR-V | ALUTs | FFs | Multipliers |"
)
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for line in open(os.path.join(HERE, "report", "split", "split_raw.txt")):
    side, rest = line.strip().split(",", 1)
    side = int(side)
    rows = runs[side]
    steps = dict(kv.split("=") for kv in rest.split(","))
    cpu = [(float(r["user_s"]) + float(r["sys_s"])) / float(r["wall_s"]) for r in rows]
    print(
        "| %dx%d | %d | %.0f%%-%.0f%% | %s | %s s | %s s | %s s | %s s | %s | %s | %s |"
        % (
            side,
            side,
            len(rows),
            100 * min(cpu),
            100 * max(cpu),
            "-".join(
                sorted({r["load_start"] for r in rows}, key=float)[
                    :: max(1, len(rows) - 1)
                ]
            ),
            steps["device_fe"],
            steps["host"],
            steps["link"],
            steps["spirv"],
            "{:,}".format(int(rows[0]["aluts"])),
            "{:,}".format(int(rows[0]["ffs"])),
            "{:,}".format(int(float(rows[0]["frac_dsps"]))),
        )
    )
