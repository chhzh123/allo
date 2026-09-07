#!/usr/bin/env python3
"""Assemble the evaluation bundle's top level from its packages.

    python3 bundle_assemble.py /scratch/hc676/spmw_eval_remaining_2026-09-06

Reads every package's own results.csv (each written by the run that produced
it), and writes at the bundle root:

  results.csv    every row of every package, with a `package` column first
  results.json   the same, plus a per-package summary of statuses
  coverage.md    what the plan asked for against what exists, per experiment
  manifest.json  the packages, their row counts, the tools and the commit

Nothing is recomputed or estimated here: a package's rows are copied as they
were written, and a cell the package left empty stays empty.
"""
import csv
import datetime
import json
import os
import subprocess
import sys

PACKAGES = [
    ("E1", "e1_gemm/spmw", "SPMW output-stationary int8 GEMM mesh (stream-fed)"),
    ("E1", "e1_gemm/spmw_mem", "SPMW GEMM, memory-fed kernel (AXI movers)"),
    ("E1", "e1_gemm/autosa", "AutoSA-generated int8 GEMM"),
    ("E1", "e1_gemm/allo", "Allo library systolic int8 GEMM"),
    ("E2", "e2_fft/spmw", "SPMW folded radix-2 FFT (delay-feedback pipeline)"),
    ("E2", "e2_fft/hpfft", "HP-FFT hand-tuned HLS radix-2 FFT"),
    ("E2", "e2_fft/allo", "Allo strided in-place FFT"),
    ("E3", "e3_tpu", "Complete GPT-2 medium and LLaMA-7B blocks on the mini-TPU"),
    ("E4", "e4_feather", "FEATHER: the SPMW port against the original RTL"),
    ("E5", "e5_compile", "Compilation time: shared vs per-instance kernels"),
    ("E6", "e6_attention", "Grouped attention-PV on a fixed workload"),
    ("E7", "e7_loc", "Language-aware source size, HLS vs SPMW"),
]

PLAN = {
    "E1": "int8 GEMM at 4/8/16/32 for SPMW, AutoSA and Allo: cosimulation cycles, routed resources, timing",
    "E2": "radix-2 FFT at 128/256/512/1024 for SPMW, HP-FFT and Allo: latency, interval, routed resources",
    "E3": "complete GPT-2 medium and LLaMA-7B blocks chained on one bitstream, with its resources and timing",
    "E4": "FEATHER GEMM and convolution with general weights, SPMW against the original RTL, plus P&R",
    "E5": "compilation time: shared/serial, shared/parallel and per-instance, three repetitions, randomised order",
    "E6": "grouped versus conventional attention on the same hardware budget and the same workload",
    "E7": "language-aware recount of design-only source size on both sides",
}

PASS = ("pass", "ok")


def read(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def main():
    root = sys.argv[1]
    everything, summary, columns = [], {}, ["package", "experiment"]
    for exp, package, title in PACKAGES:
        rows = read(os.path.join(root, package, "results.csv"))
        counts = {}
        for row in rows:
            status = (row.get("status") or "").lower()
            counts[status] = counts.get(status, 0) + 1
            merged = {"package": package, "experiment": exp}
            merged.update(row)
            everything.append(merged)
            for key in merged:
                if key not in columns:
                    columns.append(key)
        summary[package] = {
            "experiment": exp,
            "title": title,
            "rows": len(rows),
            "by_status": counts,
            "passing": sum(v for k, v in counts.items() if k in PASS),
            "has_readme": os.path.isfile(os.path.join(root, package, "README.md")),
        }

    with open(os.path.join(root, "results.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(everything)
    with open(os.path.join(root, "results.json"), "w", encoding="utf-8") as handle:
        json.dump({"packages": summary, "rows": everything}, handle, indent=1)

    commit = ""
    path = os.path.join(root, "sources", "commit.txt")
    if os.path.isfile(path):
        commit = open(path, encoding="utf-8").read().strip().splitlines()[0]
    manifest = {
        "bundle": os.path.basename(root.rstrip("/")),
        "assembled": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": subprocess.run(["hostname"], capture_output=True, text=True, check=False).stdout.strip(),
        "board": "AMD Alveo U280, xcu280-fsvh2892-2L-e",
        "tools": "Vitis HLS 2023.2, Vivado 2023.2, XRT for the board runs",
        "target": "300 MHz (3.333 ns) unless a package says otherwise",
        "allo_commit": commit,
        "packages": summary,
        "totals": {
            "rows": len(everything),
            "passing": sum(p["passing"] for p in summary.values()),
        },
    }
    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)

    lines = [
        "# Coverage: what the plan asked for, and what this bundle holds",
        "",
        f"Assembled {manifest['assembled']} on {manifest['host']}. "
        f"{manifest['totals']['rows']} rows, {manifest['totals']['passing']} of them passing. "
        "Every row is as its package wrote it; a status other than `pass`/`ok` carries its own "
        "`failure_reason`, and no value here is estimated.",
        "",
        "| Experiment | Asked for | Package | Rows | Passing | Statuses |",
        "|---|---|---|---:|---:|---|",
    ]
    for exp, package, _title in PACKAGES:
        s = summary[package]
        statuses = ", ".join(f"{k or 'blank'}={v}" for k, v in sorted(s["by_status"].items()))
        lines.append(
            f"| {exp} | {PLAN[exp]} | `{package}` | {s['rows']} | {s['passing']} | "
            f"{statuses or '--'} |"
        )
    lines += [
        "",
        "Each package directory holds its own `README.md` (what was run, with the exact commands, "
        "how each metric was measured, and its caveats), `results.csv`/`results.json`, and "
        "`reports/` with the tool reports and logs the rows were read from.",
    ]
    with open(os.path.join(root, "coverage.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    for exp, package, _t in PACKAGES:
        s = summary[package]
        print(f"  {package:22s} {s['rows']:4d} rows  {s['passing']:4d} passing  readme={s['has_readme']}")
    print(f"  {'TOTAL':22s} {manifest['totals']['rows']:4d} rows  {manifest['totals']['passing']:4d} passing")


if __name__ == "__main__":
    main()
