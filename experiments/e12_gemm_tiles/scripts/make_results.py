#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""results.csv and the README's tables, read out of the exported reports.

    python3 scripts/make_results.py            # writes results.csv, prints the tables

Every number comes from a file under a `report/` directory; a report that is
not there leaves its cells empty rather than estimated.

An SPMW array is one build a size and runs every workload, so its reports are
`<folder>/S<n>/report/` with one cycle record a workload, `W<side>_*.txt`. A
baseline is generated for its workload, so it has `<folder>/S<n>_W<side>/`.

Cycles are first memory beat to last, as in E1, wherever a waveform of the
run was kept (`beats.txt`, or SPMW's own testbench); otherwise they are the
flow's own figure from start to done, and `cycles_from` says which.  The peak
is one multiply-add a cell a cycle: `W^3 / S^2` cycles.
"""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_NS = 3.333
SIZES = (16, 32)
SIDES = (64, 128, 256, 512, 1024)
# folder, name in the tables, scope, one build a size (not a workload)
SYSTEMS = (
    ("spmw_mesh", "SPMW mesh", "array", True),
    ("spmw_kernel", "SPMW kernel", "kernel", True),
    ("dato", "Dato's design", "kernel", False),
    ("autosa_wide", "AutoSA", "kernel", False),
    ("autosa", "AutoSA, K whole", "kernel", False),
    ("allo", "Allo", "kernel", False),
)
COLUMNS = (
    "array side folder system scope port_bits_in port_bits_out ideal_cycles "
    "cycles_data_beats cycles_flow_reported cycles cycles_from pct_of_peak "
    "lut ff dsp bram_18k_equiv wns_ns achieved_ns fmax_mhz gops unrouted"
).split()


def read(*parts):
    path = os.path.join(HERE, *parts)
    if not os.path.isfile(path):
        return ""
    with open(path, errors="replace") as handle:
        return handle.read()


def cell(text, name):
    """The `Used` column of a row of report_utilization."""
    found = re.search(r"^\|\s*%s\*?\s*\|\s*([\d.]+)\s*\|" % re.escape(name), text, re.M)
    return found.group(1) if found else ""


def routed(*where):
    util = read(*where, "util.rpt")
    row = {}
    if not util:
        return row
    row["lut"] = cell(util, "CLB LUTs")
    row["ff"] = cell(util, "CLB Registers")
    row["dsp"] = cell(util, "DSPs")
    tiles = cell(util, "Block RAM Tile")
    # 18K equivalents: a tile is one RAMB36 or two RAMB18, so twice the tiles.
    row["bram_18k_equiv"] = ("%g" % (2 * float(tiles))) if tiles else ""
    found = re.search(
        r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)",
        read(*where, "timing.rpt"),
        re.S,
    )
    if found:
        wns = float(found.group(1))
        row["wns_ns"] = "%.3f" % wns
        row["achieved_ns"] = "%.3f" % (TARGET_NS - wns)
        row["fmax_mhz"] = "%d" % round(1000.0 / (TARGET_NS - wns))
    found = re.search(
        r"# of nets with routing errors\.*\s*:\s*(\d+)", read(*where, "route_status.rpt")
    )
    if found:
        row["unrouted"] = found.group(1)
    for log in ("build_pnr.log", "build.log"):
        found = re.search(r"(?:SPMW UNROUTED|PNR_UNROUTED) (\d+)", read(*where, log))
        if found:
            row["unrouted"] = found.group(1)
    return row


def ports(*where):
    """Widest read master and the write master, as synthesised."""
    ins, outs = [], []
    for line in read(*where, "port_widths.txt").splitlines():
        found = re.search(r"C_M_AXI_(\w+?)_DATA_WIDTH = (\d+)", line)
        if not found:
            continue
        # SPMW names the mover in front; the others name the argument's bundle
        # (A, B, C or gmem0, gmem1, gmem2), and the last of them is the result.
        writes = line.startswith("drain") or found.group(1).endswith(("C", "2"))
        (outs if writes else ins).append(int(found.group(2)))
    if not ins or not outs:
        return {}
    return {"port_bits_in": str(max(ins)), "port_bits_out": str(max(outs))}


def spmw_cycles(folder, size, side):
    row = {}
    if folder == "spmw_mesh":
        text = read(folder, f"S{size}", "report", f"W{side}_cosim.txt")
        found = re.search(r"SPMW CYCLES total=(\d+)", text)
        if found and "SPMW COSIM PASS" in text:
            row["cycles_data_beats"] = found.group(1)
        return row
    text = read(folder, f"S{size}", "report", f"W{side}_kernel_sim.txt")
    found = re.search(r"first_beat (\d+) last_beat (\d+) cycles (\d+)", text)
    if found and re.search(r"RESULT 0 of \d+ byte", text):
        row["cycles_data_beats"] = found.group(3)
    start = re.search(r"starting at (\d+)", text)
    done = re.search(r"done after \d+ poll\(s\) at (\d+)", text)
    period = re.search(r"clock period ([\d.]+) ns", text)
    if start and done and period:
        row["cycles_flow_reported"] = "%d" % round(
            (int(done.group(1)) - int(start.group(1))) / (1000 * float(period.group(1)))
        )
    return row


def baseline_cycles(*where):
    row = {}
    report = lambda name: read(*where, name)
    verdict = report("cosim_verdict.txt")
    if "PASS" not in verdict:
        return row
    found = re.search(r"BEATS first (\d+) last (\d+) cycles (\d+)", report("beats.txt"))
    if found:
        row["cycles_data_beats"] = found.group(3)
    for name in ("kernel0_cosim.rpt", "gemm_cosim.rpt", "top_cosim.rpt"):
        flow = re.search(
            r"\|\s*Verilog\|\s*Pass\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", report(name)
        )
        if flow:
            row["cycles_flow_reported"] = flow.group(3)
    return row


def finish(row, size, side):
    ideal = side**3 // size**2
    row["ideal_cycles"] = str(ideal)
    for source in ("beats", "flow"):
        key = "cycles_data_beats" if source == "beats" else "cycles_flow_reported"
        if row[key]:
            row["cycles"], row["cycles_from"] = row[key], source
            break
    if row["cycles"]:
        share = 100.0 * ideal / int(row["cycles"])
        # A share within a tenth of the peak keeps a second decimal, so that
        # it does not print as the peak itself.
        row["pct_of_peak"] = ("%.2f" if share >= 99.95 else "%.1f") % share
        if row["achieved_ns"]:
            # Two operations a multiply-add, as the Dato paper counts them.
            row["gops"] = "%.1f" % (
                2.0 * side**3 / (int(row["cycles"]) * float(row["achieved_ns"]))
            )
    return row


def rows():
    out = []
    for size in SIZES:
        for side in SIDES:
            for folder, name, scope, shared in SYSTEMS:
                where = (
                    (folder, f"S{size}", "report")
                    if shared
                    else (folder, f"S{size}_W{side}", "report")
                )
                if not os.path.isdir(os.path.join(HERE, *where)):
                    continue
                row = dict.fromkeys(COLUMNS, "")
                row.update(
                    array=f"{size}x{size}",
                    side=str(side),
                    folder=folder,
                    system=name,
                    scope=scope,
                )
                row.update(routed(*where))
                row.update(ports(*where))
                if shared:
                    row.update(spmw_cycles(folder, size, side))
                else:
                    row.update(baseline_cycles(*where))
                out.append(finish(row, size, side))
    return out


def num(text):
    return f"{int(float(text)):,}" if text else ""


def cycle_table(data, size):
    """A workload a row, a system a column: cycles and the share of the peak."""
    names = [
        s[1] for s in SYSTEMS if any(r["system"] == s[1] and r["cycles"] for r in data)
    ]
    lines = [
        "| GEMM | peak | " + " | ".join(names) + " |",
        "|---|---:|" + "---:|" * len(names),
    ]
    for side in SIDES:
        picked = {
            r["system"]: r
            for r in data
            if r["array"] == f"{size}x{size}" and r["side"] == str(side)
        }
        if not picked:
            continue
        cells = []
        for name in names:
            row = picked.get(name)
            if not row or not row["cycles"]:
                cells.append("")
                continue
            mark = "" if row["cycles_from"] == "beats" else "†"
            cells.append(f"{num(row['cycles'])}{mark} ({row['pct_of_peak']}%)")
        ideal = side**3 // size**2
        lines.append(f"| {side}^3 | {ideal:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def gops_table(data, size):
    """Operations a second at each design's own routed clock, in billions."""
    names = [s[1] for s in SYSTEMS if any(r["system"] == s[1] and r["gops"] for r in data)]
    lines = [
        "| GEMM | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for side in SIDES:
        picked = {
            r["system"]: r["gops"]
            for r in data
            if r["array"] == f"{size}x{size}" and r["side"] == str(side)
        }
        if any(picked.values()):
            lines.append(
                f"| {side}^3 | " + " | ".join(picked.get(n, "") for n in names) + " |"
            )
    return "\n".join(lines)


def area_table(data, size):
    """What was routed: an SPMW build once, a baseline once a workload."""
    lines = [
        "| Design | GEMM | Port in/out | LUT | FF | DSP | BRAM18 | Slack | Clock |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for folder, name, scope, shared in SYSTEMS:
        mine = [
            r
            for r in data
            if r["array"] == f"{size}x{size}" and r["folder"] == folder and r["lut"]
        ]
        for k, row in enumerate(mine[:1] if shared else mine):
            port = (
                "streams"
                if scope == "array"
                else (
                    f"{row['port_bits_in']}/{row['port_bits_out']}"
                    if row["port_bits_in"]
                    else ""
                )
            )
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    name if k == 0 else "",
                    "every one" if shared else f"{row['side']}^3",
                    port,
                    num(row["lut"]),
                    num(row["ff"]),
                    num(row["dsp"]),
                    row["bram_18k_equiv"],
                    f"{float(row['wns_ns']):+.3f} ns" if row["wns_ns"] else "",
                    f"{row['fmax_mhz']} MHz" if row["fmax_mhz"] else "",
                )
            )
    return "\n".join(lines)


def main():
    data = rows()
    with open(os.path.join(HERE, "results.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
    for size in SIZES:
        if not any(r["array"] == f"{size}x{size}" for r in data):
            continue
        print(f"### {size}x{size}\n")
        print(cycle_table(data, size))
        print()
        print(gops_table(data, size))
        print()
        print(area_table(data, size))
        print()
    bad = [r for r in data if r["unrouted"] not in ("", "0")]
    if bad:
        sys.exit(
            "unrouted nets in: "
            + ", ".join(f"{r['folder']}/{r['array']}/{r['side']}" for r in bad)
        )


if __name__ == "__main__":
    main()
