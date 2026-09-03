# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fig. 11's no-reuse ablation: what kernel reuse is actually worth.

The split backend compiles one HLS project per *role* -- per wiring class --
and instantiates it once per site. The ablation holds the RTL architecture
fixed and compiles one project per *site* instead, which is what a backend
without reuse would do. The generated C++ is identical; only the number of
`vitis_hls` invocations changes.

Both halves run under the same worker limit, so the difference is reuse and not
parallelism. Wall time and aggregate CPU time are reported separately for the
same reason.

    python3 scripts/spmw_ablate_reuse.py --design gemm8 --size 8 --jobs 40
"""

import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spmw_build_array import design, stage, synthesise  # noqa: E402
from allo.spmw.role_ip import UnitEmitter  # noqa: E402
import allo.spmw as spmw  # noqa: E402


def sites_per_role(graph):
    """Map role name -> number of sites that role covers."""
    emitter = UnitEmitter(graph)
    counts = {}
    for placement in emitter.placements():
        for order in range(len(emitter.classes(placement))):
            name = emitter.role_name(placement, order)
            _signature, _routing, sites = emitter.classes(placement)[order]
            counts[name] = len(sites)
    return counts


def replicate(out, counts):
    """One project directory per site, holding that site's role's code."""
    names = []
    for role, count in counts.items():
        src = os.path.join(out, role)
        for index in range(count):
            if index == 0:
                names.append(role)  # reuse the staged one for the first site
                continue
            clone = f"{role}__s{index}"
            dst = os.path.join(out, clone)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            os.makedirs(dst)
            for item in ("kernel.cpp", "run.tcl", f"{role}.sv"):
                source = os.path.join(src, item)
                if os.path.exists(source):
                    shutil.copy2(source, os.path.join(dst, item))
            names.append(clone)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="gemm8")
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--part", default="xcu280-fsvh2892-2L-e")
    ap.add_argument("--frequency", type=float, default=300.0)
    ap.add_argument("--jobs", type=int, default=40)
    args = ap.parse_args()

    out = (
        args.out
        or f"/scratch/{os.environ.get('USER','x')}/ablate_{args.design}_{args.size}"
    )
    os.makedirs(out, exist_ok=True)

    fab = design(args.design, args.size)
    graph = spmw.elaborate(fab)
    names = stage(graph, out, args.part, args.frequency)
    counts = sites_per_role(graph)
    instances = sum(counts.values())
    print(
        f"design={args.design} size={args.size}: "
        f"{len(names)} roles, {instances} instances, jobs={args.jobs}"
    )

    print("\n--- with reuse: one HLS project per role ---", flush=True)
    start = time.time()
    times = synthesise(out, names, jobs=args.jobs)
    reuse_wall = round(time.time() - start, 1)
    reuse_cpu = round(sum(times.values()), 1)
    print(f"REUSE   wall={reuse_wall}s cpu={reuse_cpu}s projects={len(names)}")

    print("\n--- no reuse: one HLS project per site ---", flush=True)
    clones = replicate(out, counts)
    start = time.time()
    times = synthesise(out, clones, jobs=args.jobs)
    nors_wall = round(time.time() - start, 1)
    nors_cpu = round(sum(times.values()), 1)
    print(f"NOREUSE wall={nors_wall}s cpu={nors_cpu}s projects={len(clones)}")

    print("\n=== ablation ===")
    print(
        f"  projects   {len(names):6d} -> {len(clones):6d}"
        f"  ({len(clones)/max(1,len(names)):.1f}x)"
    )
    print(
        f"  wall (s)   {reuse_wall:6.1f} -> {nors_wall:6.1f}"
        f"  ({nors_wall/max(0.1,reuse_wall):.1f}x)"
    )
    print(
        f"  cpu  (s)   {reuse_cpu:6.1f} -> {nors_cpu:6.1f}"
        f"  ({nors_cpu/max(0.1,reuse_cpu):.1f}x)"
    )
    print("ABLATION DONE")


if __name__ == "__main__":
    main()
