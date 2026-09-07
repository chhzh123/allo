# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One compilation-time measurement of one design in one mode (E5).

The split backend compiles one HLS project per *role* (wiring class) and
instantiates it once per site. This measures that against a backend without
reuse, with the RTL architecture held fixed: the generated C++ is identical
in every mode, only the number of `vitis_hls` invocations and the worker cap
change.

    --mode shared-serial     one project per role, one worker
    --mode shared-parallel   one project per role, --jobs workers
    --mode per-instance      one project per site,  --jobs workers

Every run uses a fresh directory and reports, on one `E5` line: the
frontend time (elaboration and per-role code generation), the HLS wall time,
the sum of the jobs' own elapsed times (what the old ablation called CPU),
and the user and system CPU time of the tool processes from
``getrusage(RUSAGE_CHILDREN)``, with the machine's load average at the start
and the end. A driver randomises the mode order and repeats; this script
measures one point so that a timeout can bound one mode at a time.

    python3 scripts/spmw_ablate_compile.py --design gemm8 --size 32 \\
        --mode per-instance --jobs 8 --out /scratch/$USER/e5/...
"""

import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spmw_ablate_reuse import replicate, sites_per_role  # noqa: E402
from spmw_build_array import design, stage, synthesise  # noqa: E402
import allo.spmw as spmw  # noqa: E402

MODES = ("shared-serial", "shared-parallel", "per-instance")


def _children_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime, usage.ru_stime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="gemm8")
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="xcu280-fsvh2892-2L-e")
    ap.add_argument("--frequency", type=float, default=300.0)
    args = ap.parse_args()

    if os.path.isdir(args.out) and os.listdir(args.out):
        raise SystemExit(f"{args.out} is not empty: every run gets a fresh directory")
    os.makedirs(args.out, exist_ok=True)
    load_start = os.getloadavg()[0]

    t0 = time.time()
    fab = design(args.design, args.size)
    graph = spmw.elaborate(fab)
    names = stage(graph, args.out, args.part, args.frequency)
    counts = sites_per_role(graph)
    instances = sum(counts.values())
    stage_s = round(time.time() - t0, 2)

    if args.mode == "per-instance":
        projects = replicate(args.out, counts)
        jobs = args.jobs
    else:
        projects = list(names)
        jobs = 1 if args.mode == "shared-serial" else args.jobs
    print(
        f"design={args.design} size={args.size} mode={args.mode} jobs={jobs}: "
        f"{len(names)} roles, {instances} instances, {len(projects)} projects, "
        f"frontend {stage_s}s",
        flush=True,
    )

    cpu0 = _children_cpu()
    t1 = time.time()
    times = synthesise(args.out, projects, jobs=jobs)
    hls_wall_s = round(time.time() - t1, 1)
    cpu1 = _children_cpu()
    record = {
        "design": args.design,
        "size": args.size,
        "mode": args.mode,
        "jobs": jobs,
        "roles": len(names),
        "instances": instances,
        "projects": len(projects),
        "completed_jobs": len(times),
        "frontend_s": stage_s,
        "hls_wall_s": hls_wall_s,
        "hls_sum_job_elapsed_s": round(sum(times.values()), 1),
        "cpu_user_s": round(cpu1[0] - cpu0[0], 1),
        "cpu_sys_s": round(cpu1[1] - cpu0[1], 1),
        "load_start": round(load_start, 1),
        "load_end": round(os.getloadavg()[0], 1),
        "job_elapsed_s": {k: round(v, 1) for k, v in times.items()},
    }
    with open(os.path.join(args.out, "e5.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    print(
        "E5 " + " ".join(f"{k}={v}" for k, v in record.items() if k != "job_elapsed_s"),
        flush=True,
    )


if __name__ == "__main__":
    main()
