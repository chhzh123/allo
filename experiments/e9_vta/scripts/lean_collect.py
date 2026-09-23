# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect the lean-cell builds into one table.

Each build is `scripts/spmw_build_array.py --cosim --pnr`, logged to
``logs/b_<design>_<size>.log``, with ``util_hier.rpt`` from ``hier.sh``.  The
row is the ``dut`` instance -- the harness's LFSRs are not the design, and
Gemmini and VTA are routed bare -- and the interval is the steady one, from
the done cycles of tiles 2 and 15, so neither the fill nor the drain is in it.

    python3 lean_collect.py <build root> [design_size ...]
"""

import csv
import os
import re
import sys

PERIOD = 3.333


def row(root, tag):
    log = open(os.path.join(root, "logs", f"b_{tag}.log"), errors="replace").read()
    rpt = os.path.join(root, f"b_{tag}", "util_hier.rpt")
    out = {"build": tag}
    m = re.search(r"array clock: ([\d.]+) ns achieved .* \(WNS ([+-][\d.]+) ns\)", log)
    if m:
        out["achieved_ns"] = float(m.group(1))
        out["wns_ns"] = float(m.group(2))
        out["mhz"] = round(1000.0 / float(m.group(1)), 1)
    m = re.search(r"COSIM (PASS|FAIL) \((\d+)/(\d+) tokens, (\d+) errors\)", log)
    out["cosim"] = f"{m.group(1)} {m.group(2)}/{m.group(3)}" if m else "none"
    m = re.search(r"CYCLES total=(\d+) first_out=(\d+)", log)
    if m:
        out["total_cycles"], out["first_out"] = int(m.group(1)), int(m.group(2))
    done = {int(k): int(v) for k, v in re.findall(r"XFORM (\d+) done_cycle=(\d+)", log)}
    if 2 in done and 15 in done:
        out["cycles_per_tile"] = round((done[15] - done[2]) / 13, 2)
    if os.path.isfile(rpt):
        for line in open(rpt, errors="replace"):
            cells = [c.strip() for c in line.split("|")]
            if len(cells) > 8 and cells[1] == "dut":
                out["lut"], out["ff"] = int(cells[3]), int(cells[7])
                break
    return out


def main():
    root = sys.argv[1]
    tags = sys.argv[2:] or sorted(
        f[2:-4] for f in os.listdir(os.path.join(root, "logs")) if f.startswith("b_")
    )
    fields = [
        "build",
        "cosim",
        "cycles_per_tile",
        "first_out",
        "total_cycles",
        "lut",
        "ff",
        "wns_ns",
        "achieved_ns",
        "mhz",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for tag in tags:
        writer.writerow(row(root, tag))


if __name__ == "__main__":
    main()
