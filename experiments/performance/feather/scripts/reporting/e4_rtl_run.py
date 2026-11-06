#!/usr/bin/env python3
"""E4: one run of the general FEATHER RTL bench, end to end.

    e4_rtl_run.py --rtl <RTL dir> --N 8 --out <run dir> [generator options]

generates the operands and SRAM images (e4_feather_gen.py), compiles the RTL
with tb_feather_rtl_general.sv (xvlog/xelab, cached per RTL, size and image
shape), runs xsim with the run directory as a plusarg, and checks the logged
bus rows against the model, writing <run dir>/check.json.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "e4_feather_gen.py")
TB = os.path.join(HERE, "tb_feather_rtl_general.sv")
BUILDS = "/scratch/hc676/e4_rtl_runs/builds"


def sh(cmd, cwd, log):
    with open(log, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.call(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    return rc


def build(rtl, meta, force=False):
    generics = {k: meta[k] for k in ("N", "T", "MODE", "WA", "IA", "A_BASE", "A_STRIDE", "G")}
    key = hashlib.md5((os.path.abspath(rtl) + json.dumps(generics, sort_keys=True) + open(TB, "rb").read().decode()).encode()).hexdigest()[:10]
    name = os.path.basename(os.path.abspath(rtl).rstrip("/").replace("/RTL", "")) or "rtl"
    tag = os.path.basename(os.path.dirname(os.path.abspath(rtl)))
    work = os.path.join(BUILDS, f"{tag}_N{meta['N']}_T{meta['T']}_M{meta['MODE']}_{key}")
    marker = os.path.join(work, "BUILD_OK")
    if os.path.exists(marker) and not force:
        return work, 0.0
    os.makedirs(work, exist_ok=True)
    sources = sorted(os.path.join(rtl, f) for f in os.listdir(rtl) if f.endswith(".v"))
    t0 = time.time()
    rc = sh(["xvlog", "-sv", "--relax", *sources, TB], work, os.path.join(work, "xvlog.log"))
    if rc != 0:
        sys.exit(f"xvlog failed, see {work}/xvlog.log")
    cmd = ["xelab", "tb", "-s", "sim", "--relax", "--timescale", "1ns/1ps", "-O2"]
    for k, v in generics.items():
        cmd += ["-generic_top", f"{k}={v}"]
    rc = sh(cmd, work, os.path.join(work, "xelab.log"))
    if rc != 0:
        sys.exit(f"xelab failed, see {work}/xelab.log")
    with open(marker, "w", encoding="utf-8") as f:
        f.write(json.dumps(generics))
    return work, round(time.time() - t0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--rtl", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--zpa", type=int, default=0)
    p.add_argument("--zpw", type=int, default=0)
    p.add_argument("--no-sim", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--dbg", action="store_true", help="print PE (0,0)'s internals around the first pass")
    args, gen_args = p.parse_known_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    cmd = [sys.executable, GEN, "gen", "--N", str(args.N), "--out", out, "--zpa", str(args.zpa), "--zpw", str(args.zpw), *gen_args]
    rc = sh(cmd, out, os.path.join(out, "gen.log"))
    if rc != 0:
        sys.exit(f"generation failed, see {out}/gen.log")
    with open(os.path.join(out, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    t_gen = round(time.time() - t0, 1)
    work, t_build = build(args.rtl, meta, force=args.rebuild)
    if args.no_sim:
        print(json.dumps({"gen_s": t_gen, "build_s": t_build, "build": work, "meta": meta}))
        return
    t1 = time.time()
    log = os.path.join(out, "xsim.log")
    rc = sh(["xsim", "sim", "-R", "-testplusarg", f"dir={out}", "-testplusarg", f"zpa={args.zpa}", "-testplusarg", f"zpw={args.zpw}", "-testplusarg", f"dbg={int(args.dbg)}"], work, log)
    t_sim = round(time.time() - t1, 1)
    res = subprocess.run([sys.executable, GEN, "check", "--out", out, "--log", log], capture_output=True, text=True, check=False)
    with open(os.path.join(out, "check.log"), "w", encoding="utf-8") as f:
        f.write(res.stdout + res.stderr)
    try:
        result = json.loads(res.stdout)
    except json.JSONDecodeError:
        result = {"status": "check_error", "stderr": res.stderr[-2000:]}
    result.update({"gen_s": t_gen, "build_s": t_build, "sim_s": t_sim, "xsim_rc": rc, "build": work, "rtl": os.path.abspath(args.rtl)})
    with open(os.path.join(out, "check.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=int)
    brief = {k: result.get(k) for k in ("status", "tiles", "tiles_ok", "bad_elements", "offsets", "first_output_cycles", "completion_cycles", "steady_interval_cycles", "feed_cycles_between_toggles", "toggle_count", "buffers", "host_check", "signed_equals_rtl", "sim_s")}
    print(json.dumps(brief, default=int))


if __name__ == "__main__":
    main()
