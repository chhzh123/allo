# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""make_cmp.py <runs.ndjson> <out.csv>: one row per (flow, size, jobs) with the median
and range of each stage over the repetitions, and print the markdown tables.

Stages: `frontend` is everything before HLS (TAPA: AutoSA's codegen and the kernel
fixes plus `tapa analyze`; SPMW: the command's whole wall minus its HLS wall, so
Python start-up, elaboration, code generation and the fabric RTL); `hls` is the HLS
phase's wall clock; `assembly` is TAPA's top-level RTL assembly after HLS (SPMW
writes its fabric in the frontend); `total` is the command end to end."""
import csv
import json
import statistics
import sys

FLOWS = ["tapa", "spmw_mesh_rtl", "spmw_kernel_rtl", "spmw_mesh"]
NAMES = {
    "tapa": "AutoSA -> TAPA",
    "spmw_mesh_rtl": "SPMW mesh, RTL only",
    "spmw_kernel_rtl": "SPMW kernel, RTL only",
    "spmw_mesh": "SPMW mesh as built (+ export_design)",
}

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
rows = [r for r in rows if r["rc"] == 0]
for r in rows:
    if r["flow"] == "tapa":
        r["front"] = round(r["frontend_s"] + r.get("analyze_s", 0.0), 1)
        r["asm"] = r.get("assembly_s")
    else:
        r["front"] = r.get("frontend_s")
        r["asm"] = 0.0


def stat(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, None
    return statistics.median(values), min(values), max(values)


out = []
for jobs in (8, 1):
    for size in (4, 8, 16, 32):
        for flow in FLOWS:
            sel = [
                r
                for r in rows
                if r["flow"] == flow and r["size"] == size and r["jobs"] == jobs
            ]
            if not sel:
                continue
            rec = {"flow": flow, "size": size, "jobs": jobs, "runs": len(sel)}
            for key, field in [
                ("total", "total_s"),
                ("frontend", "front"),
                ("hls", "hls_s"),
                ("assembly", "asm"),
                ("hls_jobs_sum", "hls_jobs_sum_s"),
                ("csynth_sum", "hls_csynth_sum_s"),
                ("export_sum", "hls_export_sum_s"),
            ]:
                med, lo, hi = stat([r.get(field) for r in sel])
                rec[key + "_s"] = med
                rec[key + "_min_s"] = lo
                rec[key + "_max_s"] = hi
            rec["hls_jobs"] = sel[0].get("hls_jobs")
            loads = [float(r["load_start"].split()[0]) for r in sel] + [
                float(r["load_end"].split()[0]) for r in sel
            ]
            rec["load_min"], rec["load_max"] = min(loads), max(loads)
            out.append(rec)
with open(sys.argv[2], "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(out[0]))
    writer.writeheader()
    writer.writerows(out)

fmt = lambda v: "-" if v is None else f"{v:,.1f}"
for jobs in (8, 1):
    print(f"\n{jobs} worker(s): median s (range), frontend / HLS / assembly / total")
    print("| Array | Flow | Runs | Frontend | HLS | Assembly | Total | Range |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for rec in out:
        if rec["jobs"] != jobs:
            continue
        print(
            f"| {rec['size']}x{rec['size']} | {NAMES[rec['flow']]} | {rec['runs']} | {fmt(rec['frontend_s'])} | "
            f"{fmt(rec['hls_s'])} | {fmt(rec['assembly_s'])} | {fmt(rec['total_s'])} | "
            f"{fmt(rec['total_min_s'])}-{fmt(rec['total_max_s'])} |"
        )
