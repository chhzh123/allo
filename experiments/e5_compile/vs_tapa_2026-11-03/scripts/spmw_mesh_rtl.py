# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""E5's shared-parallel measurement (scripts/spmw_ablate_compile.py) of SPMW's mesh,
with each role's `export_design` line removed: frontend + HLS to RTL only.

The assembled array reads each role's RTL from prj/sol/syn/verilog; the exported IP
is read only for the .xci of Xilinx IP cores, which the int8 mesh has none of. TAPA,
likewise, takes its tasks' RTL from synthesis and packs once at the end, so this is
the RTL-only counterpart of `tapa analyze ... synth`.

    python3 spmw_mesh_rtl.py <allo tree> --size 32 --jobs 8 --out DIR
"""
import argparse
import os
import sys
import time

TREE = sys.argv.pop(1)
sys.path.insert(0, os.path.join(TREE, "scripts"))

from spmw_build_array import design, stage, synthesise  # noqa: E402
import allo.spmw as spmw  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--design", default="gemm8")
ap.add_argument("--size", type=int, default=8)
ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("--out", required=True)
ap.add_argument("--part", default="xcu280-fsvh2892-2L-e")
ap.add_argument("--frequency", type=float, default=300.0)
args = ap.parse_args()
if os.path.isdir(args.out) and os.listdir(args.out):
    raise SystemExit(f"{args.out} is not empty: every run gets a fresh directory")
os.makedirs(args.out, exist_ok=True)

t0 = time.time()
fab = design(args.design, args.size)
graph = spmw.elaborate(fab)
names = stage(
    graph,
    args.out,
    args.part,
    args.frequency,
    pipeline_style=getattr(fab, "spmw_pipeline_style", None),
)
for name in names:
    path = os.path.join(args.out, name, "run.tcl")
    with open(path, encoding="utf-8") as handle:
        tcl = handle.read()
    assert "export_design -format ip_catalog\n" in tcl, path
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(tcl.replace("export_design -format ip_catalog\n", ""))
stage_s = round(time.time() - t0, 2)
t1 = time.time()
times = synthesise(args.out, names, jobs=args.jobs)
hls_wall_s = round(time.time() - t1, 1)
print(
    f"E5 design={args.design} size={args.size} mode=rtl-only jobs={args.jobs} "
    f"roles={len(names)} completed_jobs={len(times)} frontend_s={stage_s} "
    f"hls_wall_s={hls_wall_s} hls_sum_job_elapsed_s={round(sum(times.values()), 1)}",
    flush=True,
)
