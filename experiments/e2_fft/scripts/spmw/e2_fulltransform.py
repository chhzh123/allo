#!/usr/bin/env python3
"""Full-transform timing for SPMW and HP-FFT on one definition.

Both sides are measured first-input-beat to last-output-beat of a single
transform, and the steady interval is the median of consecutive completion
differences -- the definition HP-FFT's own harness records and which SPMW's
cosim now reports too.

The earlier E2 table divided (launch total - first output) by 32. That mixes
two reference points: it starts at the first *output* beat, so the pipeline
fill lands inside the interval, and it divides by 32 when 33 transforms are
emitted. At N=128 that turned a true 128.0 into 132.1.
"""
import json
import pathlib
import re
import statistics
import sys

SPMW_LOGS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/hc676")
HPFFT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "experiments/e2_fft/hpfft")

rows = []
for n in (128, 256, 512, 1024):
    # ---- SPMW: parse the cosim's per-transform lines --------------------
    done, first_in = [], None
    for log in sorted(SPMW_LOGS.glob(f"e2_full_N{n}.log")) + \
               sorted(SPMW_LOGS.glob(f"e2_full/N{n}/**/*.log")):
        text = log.read_text(errors="replace")
        if "SPMW XFORM" not in text:
            continue
        for m in re.finditer(r"SPMW XFORM (\d+) done_cycle=(\d+) first_in=(\d+)", text):
            done.append(int(m.group(2)))
            first_in = int(m.group(3))
        break
    if not done:
        rows.append((n, "spmw", None)); continue
    diffs = [b - a for a, b in zip(done, done[1:])]
    tail = diffs[len(diffs) // 4:]           # last 3/4, as HP-FFT defines it
    rows.append((n, "spmw", {
        "first_in": first_in,
        "full_transform_latency": done[0] - first_in,
        "steady_interval": statistics.median(tail),
        "interval_min": min(diffs), "interval_max": max(diffs),
        "transforms": len(done),
        "ideal": n, "samples_per_beat": 1,
    }))

    # ---- HP-FFT: its harness already records these -----------------------
    f = HPFFT / f"N{n}" / "report" / "cosim_events.json"
    if not f.is_file():
        rows.append((n, "hpfft", None)); continue
    d = json.loads(f.read_text())
    rows.append((n, "hpfft", {
        "first_in": d["first_input_cycle"],
        "full_transform_latency": d["first_transform_completion_cycles"],
        "steady_interval": d["steady_interval_cycles_median"],
        "interval_min": d["steady_interval_min"], "interval_max": d["steady_interval_max"],
        "transforms": d["transforms_completed"],
        "ideal": n // 2, "samples_per_beat": 2,
    }))

hdr = ("N", "system", "s/beat", "full-xform lat", "steady", "ideal", "%ideal", "samples/cyc")
print("".join(f"{h:<16}" for h in hdr))
for n, sys_, r in rows:
    if r is None:
        print(f"{n:<16}{sys_:<16}{'(not yet measured)':<16}"); continue
    pct = 100 * r["ideal"] / r["steady"]
    print(f"{n:<16}{sys_:<16}{r['samples_per_beat']:<16}"
          f"{r['full_transform_latency']:<16}{r['steady']if False else r['steady_interval']:<16}"
          f"{r['ideal']:<16}{pct:<15.1f}%{n / r['steady_interval']:>7.3f}")
