#!/usr/bin/env python3
"""Stage the folded FFT at each N and keep what the flow emits.

The E2 report bundles kept logs and reports but not the generated code, so
this re-runs the frontend only -- elaboration and per-role codegen, no HLS --
and copies out the per-role C++ and the structural RTL. It is the same
`stage()` the measured builds called, so the code is what they compiled.
"""
import os
import shutil
import sys

sys.path.insert(0, "/scratch/hc676/allo/scripts")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from spmw_build_array import design, stage  # noqa: E402
import allo.spmw as spmw  # noqa: E402

PART = "xcu280-fsvh2892-2L-e"
OUT = "/scratch/hc676/e2_organised"

for n in (128, 256, 512, 1024):
    work = f"/scratch/hc676/e2_stage/N{n}"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    fab = design("fftsdf", n)
    graph = spmw.elaborate(fab)
    names = stage(graph, work, PART, 300.0,
                  pipeline_style=getattr(fab, "spmw_pipeline_style", None))
    dest = f"{OUT}/spmw/N{n}/generated"
    os.makedirs(dest, exist_ok=True)
    kept = 0
    for root, _dirs, files in os.walk(work):
        for f in files:
            if f.endswith((".cpp", ".sv")) or f.endswith("_ROM_AUTO_1R.dat"):
                src = os.path.join(root, f)
                # one flat directory, prefixed by the role it came from
                rel = os.path.relpath(root, work).replace(os.sep, "_").strip("._")
                name = f if rel in ("", ".") else f"{rel}_{f}"
                if not os.path.exists(os.path.join(dest, name)):
                    shutil.copy2(src, os.path.join(dest, name))
                    kept += 1
    print(f"N={n}: {len(names)} roles -> {kept} files in {dest}", flush=True)
