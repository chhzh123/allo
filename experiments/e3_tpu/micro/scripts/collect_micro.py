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


def hierarchy(path):
    """The array's own LUTs, FFs and DSPs, out of the harness that wraps it.

    `--pnr` routes `spmw_harness` -- the fabric plus one LFSR per channel, so
    its stream ports stop being pins -- and Gemmini's `MxuVpu` is routed bare.
    The `dut` row is the like-for-like figure; the difference between the two is
    what the harness cost, and it is reported rather than assumed negligible.
    """
    text = read(path)
    out = {}
    for label, key in (("spmw_harness", "top"), ("dut", "dut")):
        m = re.search(
            r"^\|\s*%s\s*\|[^|]*\|\s*(\d+)\s*\|[^|]*\|[^|]*\|[^|]*\|"
            r"\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
            r"\s*(\d+)\s*\|" % label,
            text,
            re.M,
        )
        if m:
            out[key] = {
                "lut": int(m.group(1)),
                "ff": int(m.group(2)),
                "bram_36k": int(m.group(3)),
                "bram_18k": int(m.group(4)),
                "uram": int(m.group(5)),
                "dsp": int(m.group(6)),
            }
    if "dut" in out and "top" in out:
        out["harness_lut"] = out["top"]["lut"] - out["dut"]["lut"]
        out["harness_ff"] = out["top"]["ff"] - out["dut"]["ff"]
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
    row = {
        "system": "SPMW",
        "size": size,
        "variant": tag.lstrip("_") or "micro",
        "harness": "xsim on the assembled array",
    }

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

    # A variant is routed only if it is its own netlist: `noclip` reprograms the
    # `micro` one, so it has no directory here and correctly gets no area.
    out = os.path.join(root, f"spmw_pnr{tag}_S{size}")
    if os.path.isdir(out):
        row.update(utilisation(os.path.join(out, "util.rpt")))
        row.update(timing(os.path.join(out, "timing.rpt")))
        plog = read(os.path.join(root, "logs", f"spmw_pnr{tag}_S{size}.log"))
        m = re.search(r"^\s*unrouted (\d+)", plog, re.M)
        if m:
            row["unrouted"] = int(m.group(1))
        # The array without its routing harness: what compares to a bare MxuVpu.
        split = hierarchy(os.path.join(out, "util_hier.rpt"))
        if "dut" in split:
            row["lut_with_harness"] = row.get("lut")
            row["ff_with_harness"] = row.get("ff")
            row["harness_lut"] = split["harness_lut"]
            row["harness_ff"] = split["harness_ff"]
            row["lut"] = split["dut"]["lut"]
            row["ff"] = split["dut"]["ff"]
            row["dsp"] = split["dut"]["dsp"]
            row["bram_18k_equiv"] = (
                2 * split["dut"]["bram_36k"] + split["dut"]["bram_18k"]
            )
            row["uram"] = split["dut"]["uram"]
        row["ii"] = loop_ii(out)
    return row


def loop_ii(out):
    """Every loop HLS reported, with its initiation interval or its refusal.

    Both halves matter and an earlier version of this dropped one of them: a
    regex that wanted a plain integer latency matched only the pipelined loops,
    so the unpipelined instruction dispatch -- the whole reason SPMW is slow
    here -- was silently absent from the collected evidence.

    Columns are positional rather than matched: name, min, max, latency,
    achieved, target, count, pipelined.
    """
    found = {}
    if not os.path.isdir(out):
        return found
    # Which role index is the interior cell depends on the design, so the roles
    # are discovered rather than named: an earlier version hard-coded `mac_r4`
    # and `vpu_r1`, which are the programmable engine's and nothing else's.
    roles = sorted(
        d
        for d in os.listdir(out)
        if (d.startswith("mac_r") or d.startswith("vpu_r"))
        and os.path.isdir(os.path.join(out, d))
    )
    for role in roles:
        base = os.path.join(out, role, "prj", "sol", "syn", "report")
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if not name.endswith("_csynth.rpt"):
                continue
            for line in read(os.path.join(base, name)).splitlines():
                cells = [c.strip() for c in line.split("|")]
                if len(cells) < 10 or not cells[1].startswith("- "):
                    continue
                loop, latency, achieved, pipelined = (
                    cells[1][2:],
                    cells[4],
                    cells[5],
                    cells[8],
                )
                if pipelined not in ("yes", "no"):
                    continue
                found[f"{role}:{loop}"] = (
                    f"II={achieved}"
                    if pipelined == "yes"
                    else f"not pipelined (latency {latency})"
                )
    return found


def gemmini(root, size, area_csv, harness="xsim"):
    """One Gemmini point: the driver's cycles, and the area already measured.

    The area and timing are not rebuilt -- `experiments/e3_tpu/gemmini/` routed
    this exact top at this exact size already, and re-routing it would only
    introduce placer variance between the two halves of one table.

    ``harness`` picks between the two drivers. "xsim" is the emitted Verilog
    under xsim and covers all three sizes; "chiseltest" is the Chisel driver of
    record, which only finished at 4 and 8. They differ by a constant cycle --
    see the README -- and the table quotes xsim, the slower of the two.
    """
    stem = "gem_xsim" if harness == "xsim" else "gem_stream"
    log = read(os.path.join(root, "logs", f"{stem}_S{size}.log"))
    row = {
        "system": "Gemmini",
        "size": size,
        "variant": "MXU+VPU shift",
        "harness": harness,
    }
    m = re.search(
        r"MXUVPU_STREAM dim=(\d+) tiles=(\d+) shift=(\d+) correct=\s*(\w+) "
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


CSV_COLUMNS = (
    "run_id",
    "experiment_id",
    "system",
    "variant",
    "workload",
    "array_size",
    "tiles",
    "implementation_mode",
    "target_mhz",
    "status",
    "validation_pass",
    "latency_cycles",
    "steady_interval_cycles",
    "cycles_per_output_row",
    "array_busy_pct",
    "lut",
    "ff",
    "dsp",
    "bram_18k_equiv",
    "uram",
    "wns_ns",
    "achieved_ns",
    "unrouted",
    "harness",
    "notes",
)


def write_csv(path, rows):
    """The rows as a table, with the two derived columns spelled out.

    `cycles_per_output_row` and `array_busy_pct` are arithmetic on the measured
    interval, not separate measurements: `interval / S`, and `S³` MACs over the
    `interval x S²` MAC-cycles the array could have done in the same time.
    """
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            size, iv = row["size"], row.get("interval_cycles")
            tag = row["variant"]
            wns = row.get("wns_ns")
            # The two Gemmini harnesses measure the same design and must not
            # share a run_id; the earlier version gave both the same one, which
            # silently makes one row look like a duplicate of the other.
            harness = str(row.get("harness", ""))
            short = "chiseltest" if harness.startswith("chiseltest") else "xsim"
            slug = tag.replace("+", "").replace(" ", "_").lower()
            writer.writerow(
                {
                    "run_id": (
                        f"e3micro_{row['system'].lower()}_{slug}_{short}_S{size}"
                    ),
                    "experiment_id": "E3-micro",
                    "system": row["system"],
                    "variant": tag,
                    "workload": (
                        f"{size}x{size}x{size} int8 GEMM, bias, requantise, "
                        "ReLU, clip to int8; 16 tiles back to back"
                        + (", clip removed" if tag == "noclip" else "")
                        + (
                            "; fixed-function datapath"
                            if tag in ("fixed", "slice")
                            else ""
                        )
                        + ("; depth-2 links" if tag == "slice" else "")
                    ),
                    "array_size": f"{size}x{size}",
                    "tiles": row.get("tiles"),
                    "implementation_mode": (
                        "chisel_interpreter" if short == "chiseltest" else "rtl_sim"
                    ),
                    "target_mhz": 300,
                    "status": "pass" if row.get("validation") == "pass" else "fail",
                    "validation_pass": row.get("validation") == "pass",
                    "latency_cycles": row.get("latency_cycles"),
                    "steady_interval_cycles": iv,
                    "cycles_per_output_row": None if not iv else round(iv / size, 3),
                    "array_busy_pct": (None if not iv else round(100.0 * size / iv, 1)),
                    "lut": row.get("lut"),
                    "ff": row.get("ff"),
                    "dsp": row.get("dsp"),
                    "bram_18k_equiv": row.get("bram_18k_equiv"),
                    "uram": row.get("uram"),
                    "wns_ns": wns,
                    "achieved_ns": None if wns is None else round(3.333 - wns, 3),
                    "unrouted": row.get("unrouted"),
                    "harness": row.get("harness"),
                    "notes": row.get("tokens", ""),
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/hc676/e3_micro")
    parser.add_argument("--area-csv", required=True, help="gemmini/results.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", help="also write results.csv")
    args = parser.parse_args()

    rows = []
    for size in SIZES:
        rows.append(gemmini(args.root, size, args.area_csv))
        cross = gemmini(args.root, size, args.area_csv, "chiseltest")
        if cross.get("latency_cycles") is not None:
            rows.append(cross)
        rows.append(spmw(args.root, size))
        rows.append(spmw(args.root, size, tag="_noclip"))
        rows.append(spmw(args.root, size, tag="_fixed"))
        rows.append(spmw(args.root, size, tag="_slice"))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=1, sort_keys=True)
    if args.csv:
        write_csv(args.csv, rows)
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
