# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build `head16/results.csv` from the runs' logs and cost records.

    python3 make_results.py experiments/e6_attention/head16
"""

import csv
import json
import os
import re
import sys

root = sys.argv[1]  # experiments/e6_attention/head16
rows = []
for seq in (64, 128, 256, 512, 1024):
    for kind in ("grouped", "conv"):
        log = os.path.join(root, "logs", f"{kind}_{seq}.log")
        if not os.path.exists(log):
            continue
        text = open(log).read()
        m = re.search(r"SPMW CYCLES total=(-?\d+) first_out=(\d+) first_in=(\d+)", text)
        ok = "SPMW COSIM PASS" in text
        cost = {}
        cj = os.path.join(root, "report", f"{kind}_{seq}_cost.json")
        if os.path.exists(cj):
            cost = json.load(open(cj))
        keys = 32 if kind == "grouped" else 16
        tiles = seq // keys
        cs = cost.get("cosim_seconds", {})
        rows.append(
            {
                "array": "16x16",
                "pes": 256,
                "D": 8,
                "scheme": "grouped G=2" if kind == "grouped" else "ungrouped",
                "S": seq,
                "keys_a_tile": keys,
                "tiles": tiles,
                "rows_streamed": tiles * seq,
                "cycles": int(m.group(1)) if m else "",
                "first_output": int(m.group(2)) if m else "",
                "cosim_pass": ok,
                "hls_wall_s": cost.get("hls_wall_seconds", ""),
                "vivado_s": cost.get("vivado_seconds", ""),
                "xsim_s": cs.get("xsim", ""),
                "cosim_s": round(sum(cs.values()), 1) if cs else "",
                "design": f"attnhead16d8-{kind}-{seq}",
            }
        )
with open(os.path.join(root, "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r["scheme"], r["S"], r["cycles"], r["first_output"], r["cosim_pass"])
