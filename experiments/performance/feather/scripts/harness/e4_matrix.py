#!/usr/bin/env python3
"""E4: a set of general-bench runs on one RTL at one size, sequentially.

    e4_matrix.py --rtl <RTL dir> --N 8 --tag fixed --set validate|orig|shakedown --out <root>

Each run goes through e4_rtl_run.py; the brief results are collected in
<root>/matrix_<tag>_N<N>_<set>.json as they finish.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "e4_rtl_run.py")

ZPS = [(0, 0), (7, 5), (128, 128)]
SEEDS = [0, 1, 2]


def configs(which, N):
    out = []
    if which == "validate":
        programs = ["conv0", "conv1", "conv2", "conv3", "gemm", "random"] if N == 4 else ["gemm", "random"]
        for prog, pat, (zpa, zpw), seed in itertools.product(programs, ["small", "full", "sparse"], ZPS, SEEDS):
            out.append((f"tile_{prog}_{pat}_zp{zpa}-{zpw}_s{seed}",
                        ["--workload", "tile", "--program", prog, "--pattern", pat, "--zpa", str(zpa), "--zpw", str(zpw), "--seed", str(seed), "--mode", "1"]))
    elif which == "orig":
        for seed in SEEDS:
            out.append((f"tile_gemm_small_zp0-0_s{seed}",
                        ["--workload", "tile", "--program", "gemm", "--pattern", "small", "--seed", str(seed), "--mode", "1"]))
        out.append(("tile_gemm_small_zp7-5_s0", ["--workload", "tile", "--program", "gemm", "--pattern", "small", "--zpa", "7", "--zpw", "5", "--seed", "0", "--mode", "1"]))
    elif which == "shakedown":
        out.append(("m0_T4_mixed", ["--workload", "tile", "--tiles", "4", "--program", "mixed", "--pattern", "small", "--zpa", "7", "--zpw", "5", "--seed", "0", "--mode", "0"]))
        out.append(("m1_T4_mixed", ["--workload", "tile", "--tiles", "4", "--program", "mixed", "--pattern", "small", "--zpa", "7", "--zpw", "5", "--seed", "0", "--mode", "1"]))
        out.append(("m0_T40_mixed_full", ["--workload", "tile", "--tiles", "40", "--program", "mixed", "--pattern", "full", "--zpa", "3", "--zpw", "9", "--seed", "1", "--mode", "0"]))
        out.append(("m1_T40_mixed_full", ["--workload", "tile", "--tiles", "40", "--program", "mixed", "--pattern", "full", "--zpa", "3", "--zpw", "9", "--seed", "1", "--mode", "1"]))
    else:
        raise SystemExit(f"unknown set {which}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rtl", required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--set", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", default="", help="run only the configs whose name contains this")
    args = p.parse_args()
    root = os.path.join(args.out, f"{args.tag}_N{args.N}_{args.set}")
    os.makedirs(root, exist_ok=True)
    summary_path = os.path.join(args.out, f"matrix_{args.tag}_N{args.N}_{args.set}.json")
    results = {}
    if os.path.exists(summary_path):
        results = json.load(open(summary_path, encoding="utf-8")).get("runs", {})
    t0 = time.time()
    for name, extra in configs(args.set, args.N):
        if args.only and args.only not in name:
            continue
        run_dir = os.path.join(root, name)
        cmd = [sys.executable, RUN, "--rtl", args.rtl, "--N", str(args.N), "--out", run_dir, *extra]
        t1 = time.time()
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
        try:
            brief = json.loads(done.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            brief = {"status": "run_error", "stderr": done.stderr[-1500:], "stdout": done.stdout[-500:]}
        brief["wall_s"] = round(time.time() - t1, 1)
        results[name] = brief
        print(name, json.dumps(brief, default=int)[:400], flush=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"rtl": args.rtl, "N": args.N, "set": args.set, "runs": results, "elapsed_s": round(time.time() - t0, 1)}, f, indent=1, default=int)
    statuses = [r.get("status") for r in results.values()]
    print(f"MATRIX DONE {args.tag} N={args.N} {args.set}: {statuses.count('pass')} pass / {len(statuses)} in {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
