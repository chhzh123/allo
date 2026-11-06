# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""collect_routes.py: every finished whole-engine route as JSON.

Per design the routes at each clock target, and the one with the best clock.
A Gemmini route also carries its split by block and, where scope_timing.sh
has run, the clock of its execute scope; a VTA route the clock of its
datapath + scratchpads scope.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import split


def tagged(path, tag):
    """The value a scope-timing log printed under `tag`."""
    if not os.path.exists(path):
        return None
    hit = re.search(rf"^{tag} (\S+)", open(path, errors="replace").read(), re.M)
    return hit.group(1) if hit else None


def read(d):
    util = open(d + "/util.rpt", errors="replace").read()

    def cell(name):
        m = re.search(rf"^\|\s*{re.escape(name)}\s*\|\s*([\d.]+)", util, re.M)
        return float(m.group(1)) if m else 0.0

    tcl = open(d + "/pnr.tcl").read()
    target = float(re.search(r"create_clock -period ([\d.]+)", tcl).group(1))
    timing = open(d + "/timing.rpt", errors="replace").read()
    wns = float(re.search(r"WNS\(ns\).*?\n.*?\n\s*(-?[\d.]+)", timing, re.S).group(1))
    run = dict(
        dir=os.path.basename(d),
        target=target,
        wns=wns,
        period=round(target - wns, 3),
        lut=int(cell("CLB LUTs")),
        ff=int(cell("CLB Registers")),
        bram=cell("Block RAM Tile"),
        uram=int(cell("URAM")),
        dsp=int(cell("DSPs")),
    )
    for name in ("scope.txt", "scope.log"):
        slack = tagged(d + "/" + name, "SCOPE_SLACK")
        if slack is not None:
            run["scope_period"] = round(target - float(slack), 3)
            run["scope_from"] = tagged(d + "/" + name, "SCOPE_FROM")
            run["scope_to"] = tagged(d + "/" + name, "SCOPE_TO")
            break
    return run


out = {"gem": {}, "vta": {}}
for d in sorted(glob.glob("/scratch/hc676/gemmini_full/pnr_*")):
    if (
        not os.path.isdir(d)
        or not os.path.exists(d + "/DONE")
        or "rc=0" not in open(d + "/DONE").read()
    ):
        continue
    parts = os.path.basename(d).split("_")  # pnr, variant, dim[, tag]
    run = read(d)
    whole, rows = split.blocks(d)
    run["blocks"] = {k.strip(): v for k, v in rows.items()}
    run["scope"] = split.scope(rows)
    out["gem"].setdefault(parts[1], {}).setdefault(parts[2], {"runs": []})[
        "runs"
    ].append(run)
for w, base in ((4, "pnr_Core_w4"), (8, "pnr_Core_w8"), (16, "pnr_Core_w16")):
    for d in sorted(glob.glob(f"/scratch/hc676/vta_build/{base}*")):
        if os.path.exists(d + "/DONE") and "rc=0" in open(d + "/DONE").read():
            out["vta"].setdefault(str(w), {"runs": []})["runs"].append(read(d))
for group in (out["vta"], *out["gem"].values()):
    for entry in group.values():
        b = min(entry["runs"], key=lambda r: r["period"])
        entry.update(
            {k: b[k] for k in ("lut", "ff", "bram", "uram", "dsp")}, best=b["dir"]
        )
        scoped = [r for r in entry["runs"] if "scope_period" in r]
        if scoped:
            entry["scope_best"] = min(scoped, key=lambda r: r["scope_period"])["dir"]
print(json.dumps(out, indent=1))
