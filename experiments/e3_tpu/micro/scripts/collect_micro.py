# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read the E3 microbenchmark's measurements off the machine that made them.

One row per (system, size). Cycles come from the simulations, area and timing
from the routed reports; nothing here is computed from a model.

BRAM is reported as **RAMB18 equivalents**, `2 x Block RAM Tile`. A tile is
36 Kb, so reading a `RAMB18E2` line instead understates any design that uses
RAMB36 -- which has caught this project three times.

Run on brg-zhang-xcel::

    python3 collect_micro.py --root /scratch/hc676/e3_micro --out micro.json
"""

import argparse
import csv
import json
import os
import re
import sys

SIZES = (4, 8, 16)


def read(path):
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def utilisation(path):
    """LUT, FF, DSP and BRAM out of a `report_utilization` table."""
    text = read(path)
    if not text:
        return {}

    def cell(name):
        # Vivado prints `CLB LUTs*` with a footnote asterisk, and tiles come in
        # halves, so neither the asterisk nor the decimal point is optional.
        m = re.search(r"^\|\s*%s\*?\s*\|\s*([\d.]+)\s*\|" % re.escape(name), text, re.M)
        if not m:
            return None
        return float(m.group(1)) if "." in m.group(1) else int(m.group(1))

    tiles = cell("Block RAM Tile")
    out = {
        "lut": cell("CLB LUTs"),
        "ff": cell("CLB Registers"),
        "dsp": cell("DSPs"),
        "uram": cell("URAM"),
        "bram_tiles": tiles,
        "bram_18k_equiv": None if tiles is None else int(2 * tiles),
    }
    # The RAMB rows are the independent check on the tile row: they must agree.
    m36 = re.search(r"^\|\s*RAMB36E2\s*\|\s*(\d+)", text, re.M)
    m18 = re.search(r"^\|\s*RAMB18E2\s*\|\s*(\d+)", text, re.M)
    if m36 or m18:
        out["bram_18k_from_ramb"] = 2 * int(m36.group(1) if m36 else 0) + int(
            m18.group(1) if m18 else 0
        )
    return out


def timing(path):
    text = read(path)
    m = re.search(
        r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)\s+(-?[\d.]+)", text, re.S
    )
    if not m:
        return {}
    return {"wns_ns": float(m.group(1)), "tns_ns": float(m.group(2))}


def steady(done):
    """The median gap between tile completions, over the last three quarters.

    The first gaps carry the array's fill; the median over the tail is the
    interval a long stream would run at. E2 reads its FFT intervals the same
    way, so the two experiments' "interval" columns mean the same thing.
    """
    gaps = [b - a for a, b in zip(done, done[1:])]
    if not gaps:
        return None
    tail = sorted(gaps[len(gaps) // 4 :])
    return tail[len(tail) // 2]


def spmw(root, size, tag=""):
    """One SPMW point: the cosim's cycles and the routed run's area."""
    name = f"spmw_cosim{tag}_S{size}"
    log = read(os.path.join(root, "logs", f"{name}.log"))
    row = {"system": "SPMW", "size": size, "variant": tag.lstrip("_") or "micro"}

    m = re.search(r"SPMW COSIM (PASS|FAIL) \((\d+)/(\d+) tokens, (\d+) errors\)", log)
    if m:
        row["validation"] = m.group(1).lower()
        row["tokens"] = f"{m.group(2)}/{m.group(3)}"
        row["errors"] = int(m.group(4))
    elif "SPMW COSIM TIMEOUT" in log:
        row["validation"] = "timeout"

    xf = [
        (int(a), int(b), int(c))
        for a, b, c in re.findall(
            r"SPMW XFORM (\d+) done_cycle=(\d+) first_in=(\d+)", log
        )
    ]
    # De-duplicate: the driver echoes xsim's stdout, so each line appears twice.
    seen, done, first_in = set(), [], None
    for k, cycle, fin in xf:
        if k in seen:
            continue
        seen.add(k)
        done.append(cycle)
        first_in = fin
    if done:
        row["tiles"] = len(done)
        row["first_in"] = first_in
        row["latency_cycles"] = done[0] - first_in + 1
        row["interval_cycles"] = steady(done)
        row["last_out"] = done[-1]

    if not tag:  # the ablation shares the netlist, so it has no route of its own
        out = os.path.join(root, f"spmw_pnr_S{size}")
        row.update(utilisation(os.path.join(out, "util.rpt")))
        row.update(timing(os.path.join(out, "timing.rpt")))
        plog = read(os.path.join(root, "logs", f"spmw_pnr_S{size}.log"))
        m = re.search(r"SPMW UNROUTED (\d+)", plog)
        if m:
            row["unrouted"] = int(m.group(1))
        row["ii"] = loop_ii(out, size)
    return row


def loop_ii(out, size):
    """The achieved initiation interval of the loops HLS did pipeline.

    Quoted, not inferred: the sweep loop in the matrix cell and the accumulate
    loop in the vector lane both say `achieved 1` in their own csynth report,
    and the instruction-dispatch loops above them say `Pipelined: no`.
    """
    found = {}
    for role in ("mac_r4", "vpu_r1"):  # one matrix cell, one vector lane
        base = os.path.join(out, role, "prj", "sol", "syn", "report")
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if not name.endswith("_csynth.rpt"):
                continue
            text = read(os.path.join(base, name))
            for line in text.splitlines():
                m = re.match(
                    r"\s*\|-?\s*(\S+)\s*\|\s*\d+\|\s*\d+\|\s*(\d+)\|\s*(\d+)\|"
                    r"\s*(\d+)\|.*\|\s*(yes|no)\|",
                    line,
                )
                if m and m.group(5) == "yes":
                    found.setdefault(f"{role}:{m.group(1)}", int(m.group(3)))
                elif m:
                    found.setdefault(f"{role}:{m.group(1)}", "not pipelined")
    return found


def gemmini(root, size, area_csv):
    """One Gemmini point: the driver's cycles, and the area already measured.

    The area and timing are not rebuilt -- `experiments/e3_tpu/gemmini/` routed
    this exact top at this exact size already, and re-routing it would only
    introduce placer variance between the two halves of one table.
    """
    log = read(os.path.join(root, "logs", f"gem_stream_S{size}.log"))
    row = {"system": "Gemmini", "size": size, "variant": "MXU+VPU shift"}
    m = re.search(
        r"MXUVPU_STREAM dim=(\d+) tiles=(\d+) shift=(\d+) correct=(\w+) "
        r"rows_out=(\d+) want_rows=(\d+) wrong_rows=(\d+) "
        r"first_in=(-?\d+) last_out=(-?\d+) latency=(-?\d+) interval=(-?\d+)",
        log,
    )
    if m:
        row.update(
            {
                "tiles": int(m.group(2)),
                "validation": "pass" if m.group(4) == "true" else "fail",
                "errors": int(m.group(7)),
                "first_in": int(m.group(8)),
                "last_out": int(m.group(9)),
                "latency_cycles": int(m.group(10)),
                "interval_cycles": int(m.group(11)),
            }
        )
    for line in csv.DictReader(open(area_csv, encoding="utf-8")):
        if (
            line["array"] == f"{size}x{size}"
            and line["design"] == "MXU + VPU"
            and (line["scale_form"] == "shift")
        ):
            tiles = float(line["bram_tiles"])
            row.update(
                {
                    "lut": int(line["lut"]),
                    "ff": int(line["ff"]),
                    "dsp": int(line["dsp"]),
                    "bram_tiles": tiles,
                    "bram_18k_equiv": int(2 * tiles),
                    "uram": 0,
                    "wns_ns": float(line["wns_ns"]),
                    "unrouted": int(line["unrouted"]),
                }
            )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/hc676/e3_micro")
    parser.add_argument("--area-csv", required=True, help="gemmini/results.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for size in SIZES:
        rows.append(gemmini(args.root, size, args.area_csv))
        rows.append(spmw(args.root, size))
        rows.append(spmw(args.root, size, tag="_noclip"))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=1, sort_keys=True)
    for row in rows:
        print(
            f"{row['system']:8s} S={row['size']:<3d} {row['variant']:<16s} "
            f"valid={row.get('validation','-'):8s} "
            f"lat={str(row.get('latency_cycles','-')):>6s} "
            f"int={str(row.get('interval_cycles','-')):>6s} "
            f"lut={str(row.get('lut','-')):>7s} ff={str(row.get('ff','-')):>7s} "
            f"dsp={str(row.get('dsp','-')):>4s} "
            f"bram18={str(row.get('bram_18k_equiv','-')):>4s} "
            f"wns={str(row.get('wns_ns','-')):>7s} "
            f"unrouted={str(row.get('unrouted','-')):>3s}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
