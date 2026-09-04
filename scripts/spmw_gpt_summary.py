# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn the board walker's output into the per-stage table and the layer total.

    python3 scripts/spmw_gpt_summary.py /scratch/$USER/board_v1.log [more logs]

Reads every `spmw_board_gpt.py` log given, keeps the per-shape launch timings
and the per-stage rows, and prints one table per log plus a JSON blob the
report is built from. Purely a parser: the numbers are the walker's.
"""

import json
import re
import sys

SHAPE = re.compile(
    r"^\s+(\w+)\s+counts=(\S+)\s+(matches|MISMATCH)\s+\|\s+([\d.]+) us/launch\s+\|"
    r"\s+([\d.]+) us of array work\s+\|\s+([\d.]+)% busy"
)
STAGE = re.compile(r"^\s+(.+?)\s{2,}(\w+|-)\s+(\d+|-)\s+([\d.]+|-)\s+(\S+)\s+(\S+)\s*$")
TOTAL = re.compile(r"=> ([\d.]+) ms/layer on the accelerator, ([\d.]+) s for (\d+) layers")
PEAK = re.compile(r"=> ([\d.]+)% of the array's ([\d.]+) GMAC/s")


def parse(path):
    out = {"log": path, "shapes": {}, "stages": [], "layer_ms": None}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            m = SHAPE.match(line)
            if m:
                out["shapes"][m.group(1)] = {
                    "counts": m.group(2),
                    "correct": m.group(3) == "matches",
                    "us_per_launch": float(m.group(4)),
                    "us_array": float(m.group(5)),
                    "busy_pct": float(m.group(6)),
                }
                continue
            m = STAGE.match(line)
            if m and m.group(2) != "shape" and m.group(3) not in ("launches",):
                try:
                    launches = int(m.group(3)) if m.group(3) != "-" else None
                    ms = float(m.group(4)) if m.group(4) != "-" else None
                except ValueError:
                    continue
                out["stages"].append(
                    {"stage": m.group(1).strip(), "shape": m.group(2),
                     "launches": launches, "ms": ms,
                     "mmac": m.group(5), "gmacs": m.group(6)}
                )
                continue
            m = TOTAL.search(line)
            if m:
                out["layer_ms"] = float(m.group(1))
                out["model_s"] = float(m.group(2))
                out["layers"] = int(m.group(3))
            m = PEAK.search(line)
            if m:
                out["peak_pct"] = float(m.group(1))
                out["peak_gmacs"] = float(m.group(2))
    return out


def main():
    results = [parse(p) for p in sys.argv[1:]]
    for r in results:
        print(f"== {r['log']} ==")
        for name, s in r["shapes"].items():
            print(f"  {name:6s} {s['us_per_launch']:8.1f} us/launch  {s['busy_pct']:5.1f}% busy  "
                  f"{'ok' if s['correct'] else 'MISMATCH'}")
        for s in r["stages"]:
            if s["ms"] is not None:
                print(f"  {s['stage']:22s} {s['shape']:6s} {s['launches']:5d} launches  {s['ms']:9.3f} ms")
        if r["layer_ms"] is not None:
            print(f"  layer: {r['layer_ms']:.1f} ms  model: {r['model_s']:.2f} s  "
                  f"peak: {r.get('peak_pct', 0):.1f}%")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
