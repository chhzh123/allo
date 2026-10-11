#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""results.csv and the README's tables, read out of the exported reports.

    python3 scripts/make_results.py            # writes results.csv, prints the tables

Definitions are E2's (`experiments/e2_fft/README.md`): latency is a transform's
last output beat minus the launch's first input beat; interval is the median of
the last three quarters of the differences between consecutive completions;
BRAM18 is twice the block RAM tiles.  Two rows were measured in E2 and are read
from its `results.csv` rather than built again: HP-FFT UF1 and Allo.  The
Spiral rows are read from the reports of the SwizzleFFT project's runs, copied
under `spiral/`.
"""
import csv
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E2 = os.path.join(os.path.dirname(HERE), "e2_fft", "results.csv")
N = 1024
TARGET_NS = 3.333
COLUMNS = (
    "n framework config samples_per_cycle transforms full_transform_latency "
    "steady_interval_cycles interval_min interval_max ideal_interval_cycles "
    "pct_of_ideal lut ff dsp bram_18k_equiv wns_ns achieved_ns fmax_mhz "
    "transforms_per_s source"
).split()


def read(*parts):
    path = os.path.join(HERE, *parts)
    if not os.path.isfile(path):
        return ""
    with open(path, errors="replace") as handle:
        return handle.read()


def cell(text, name):
    found = re.search(r"^\|\s*%s\*?\s*\|\s*([\d.]+)\s*\|" % re.escape(name), text, re.M)
    return found.group(1) if found else ""


def routed(util, timing):
    row = {}
    if not util:
        return row
    row["lut"] = cell(util, "CLB LUTs")
    row["ff"] = cell(util, "CLB Registers")
    row["dsp"] = cell(util, "DSPs")
    tiles = cell(util, "Block RAM Tile")
    row["bram_18k_equiv"] = ("%g" % (2 * float(tiles))) if tiles else ""
    found = re.search(
        r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)", timing, re.S
    )
    if found:
        wns = float(found.group(1))
        row["wns_ns"] = "%.3f" % wns
        row["achieved_ns"] = "%.3f" % (TARGET_NS - wns)
        row["fmax_mhz"] = "%d" % round(1000.0 / (TARGET_NS - wns))
    return row


def finish(row):
    """Ideal interval, share of it, and transforms a second at the routed clock."""
    if row.get("samples_per_cycle"):
        ideal = N / int(row["samples_per_cycle"])
        row["ideal_interval_cycles"] = "%g" % ideal
        if row.get("steady_interval_cycles"):
            row["pct_of_ideal"] = "%.1f" % (
                100.0 * ideal / float(row["steady_interval_cycles"])
            )
    if row.get("steady_interval_cycles") and row.get("achieved_ns"):
        row["transforms_per_s"] = "%d" % round(
            1e9 / (float(row["achieved_ns"]) * float(row["steady_interval_cycles"]))
        )
    return row


def spmw(lanes):
    base = ("spmw", f"W{lanes}", "report")
    text = read(*base, "xform.txt")
    row = dict.fromkeys(COLUMNS, "")
    row.update(
        n=N,
        framework="spmw",
        config=f"lanes W={lanes}",
        samples_per_cycle=lanes,
        source=f"spmw/W{lanes}/report",
    )
    done = [
        int(m.group(2)) for m in re.finditer(r"SPMW XFORM (\d+) done_cycle=(\d+)", text)
    ]
    first = re.search(r"first_in=(\d+)", text)
    if done and first and "SPMW COSIM PASS" in text:
        diffs = [b - a for a, b in zip(done, done[1:])]
        tail = diffs[len(diffs) // 4 :]
        row.update(
            transforms=len(done),
            full_transform_latency=done[0] - int(first.group(1)),
            steady_interval_cycles="%g" % statistics.median(tail),
            interval_min=min(diffs),
            interval_max=max(diffs),
        )
    row.update(routed(read(*base, "util.rpt"), read(*base, "timing.rpt")))
    return finish(row)


def hpfft(config, unroll, label):
    base = ("hpfft", config, "report")
    row = dict.fromkeys(COLUMNS, "")
    row.update(
        n=N,
        framework="hpfft",
        config=label,
        samples_per_cycle=2 * unroll,
        source=f"hpfft/{config}/report",
    )
    events = read(*base, "cosim_events.json")
    checked = read(*base, "validate_rtl_vcd.json")
    if events and (not checked or json.loads(checked).get("validation_pass")):
        d = json.loads(events)
        row.update(
            transforms=d["transforms_completed"],
            full_transform_latency=d["first_transform_completion_cycles"],
            steady_interval_cycles="%g" % d["steady_interval_cycles_median"],
            interval_min=d["steady_interval_min"],
            interval_max=d["steady_interval_max"],
        )
    row.update(routed(read(*base, "pnr_util.rpt"), read(*base, "pnr_timing.rpt")))
    return finish(row)


def spiral(lanes):
    """A core of the Spiral DFT generator, from SwizzleFFT's run of it."""
    base = ("spiral", f"W{lanes}", "report")
    row = dict.fromkeys(COLUMNS, "")
    row.update(
        n=N,
        framework="spiral",
        config=f"streaming, radix 2, W={lanes}",
        samples_per_cycle=lanes,
        source=f"spiral/W{lanes}/report",
    )
    sim = read(*base, "sim.json")
    if sim and json.loads(sim).get("validation_pass"):
        d = json.loads(sim)
        row.update(
            transforms=d["transforms"],
            full_transform_latency=d["latency"],
            steady_interval_cycles="%g" % d["interval_median"],
            interval_min=d["interval_min"],
            interval_max=d["interval_max"],
        )
    row.update(routed(read(*base, "pnr_util.rpt"), read(*base, "pnr_timing.rpt")))
    return finish(row)


def from_e2(framework, label, samples):
    """A row E2 already holds, on E2's definitions."""
    with open(E2, newline="") as handle:
        for line in csv.DictReader(handle):
            if line["n"] == str(N) and line["framework"] == framework:
                row = dict.fromkeys(COLUMNS, "")
                row.update(
                    n=N,
                    framework=framework,
                    config=label,
                    samples_per_cycle=samples or "",
                    transforms=line["transforms"],
                    full_transform_latency=line["full_transform_latency"],
                    steady_interval_cycles="%g" % float(line["steady_interval_cycles"]),
                    source="e2_fft/results.csv",
                )
                for key in (
                    "lut",
                    "ff",
                    "dsp",
                    "bram_18k_equiv",
                    "wns_ns",
                    "achieved_ns",
                    "fmax_mhz",
                ):
                    row[key] = line[key]
                return finish(row)
    return None


def num(text):
    return f"{int(float(text)):,}" if text not in ("", None) else ""


def table(pairs):
    lines = [
        "| samples/cyc | system | latency | interval | of ideal | LUT | FF | DSP | BRAM18 "
        "| slack | MHz | transforms/s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for samples, group in pairs:
        for k, (name, row) in enumerate(group):
            if row is None:
                continue
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    samples if k == 0 else "",
                    name,
                    num(row["full_transform_latency"]),
                    (
                        f"{float(row['steady_interval_cycles']):,.1f}"
                        if row["steady_interval_cycles"]
                        else ""
                    ),
                    (row["pct_of_ideal"] + "%") if row["pct_of_ideal"] else "",
                    num(row["lut"]),
                    num(row["ff"]),
                    num(row["dsp"]),
                    row["bram_18k_equiv"],
                    ("%+.3f" % float(row["wns_ns"])) if row["wns_ns"] else "",
                    row["fmax_mhz"],
                    num(row["transforms_per_s"]),
                )
            )
    return "\n".join(lines)


def main():
    hp = {
        2: from_e2("hpfft", "UF1", 2),
        4: hpfft("UF2", 2, "UF2"),
        8: hpfft("UF4_a8", 4, "UF4, static + function pipeline"),
        16: hpfft("UF8_b4", 8, "UF8, data_2 repartitioned"),
    }
    sp = {
        w: spmw(w)
        for w in (2, 4, 8, 16)
        if os.path.isdir(os.path.join(HERE, "spmw", f"W{w}"))
    }
    sr = {
        w: spiral(w)
        for w in (2, 4, 8, 16)
        if os.path.isdir(os.path.join(HERE, "spiral", f"W{w}"))
    }
    allo = from_e2("allo", "strided, one transform a call", None)
    # E2's fallback for a UF4 that would not close: `static` alone.
    fallback = hpfft("UF4_a1", 4, "UF4, static only")
    rows = [r for w in (2, 4, 8, 16) for r in (hp.get(w), sr.get(w), sp.get(w)) if r]
    rows += [r for r in (fallback, allo) if r and r["full_transform_latency"]]
    with open(os.path.join(HERE, "results.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    names = {2: "UF1", 4: "UF2", 8: "UF4", 16: "UF8"}
    print(
        table(
            [
                (
                    w,
                    [
                        (f"HP-FFT {names[w]}", hp.get(w)),
                        (f"Spiral W={w}", sr.get(w)),
                        (f"SPMW lanes W={w}", sp.get(w)),
                    ],
                )
                for w in (2, 4, 8, 16)
            ]
        )
    )
    if allo:
        print()
        print(table([("", [("Allo, strided", allo)])]))


if __name__ == "__main__":
    main()
