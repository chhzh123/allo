#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""results.csv and the README's tables, read out of the exported reports.

    python3 scripts/make_results.py            # writes results.csv, prints the tables

Every number comes from a file under `<framework>/S<n>/report/`; a report that
is not there leaves its cells empty rather than estimated.  The columns are
E1's (`experiments/e1_gemm/results.csv`), so the two tables can be set side by
side.
"""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_NS = 3.333
SIZES = (4, 8, 16, 32)
# folder, name in the table, scope, how its cycles were measured
SYSTEMS = (
    ("spmw_mesh", "SPMW mesh", "array", "tokens"),
    ("spmw_kernel", "SPMW kernel", "kernel", "spmw_tb"),
    ("autosa", "AutoSA", "kernel", "beats"),
    ("autosa_wide", "AutoSA, wide", "kernel", "beats"),
    ("allo", "Allo", "kernel", "beats"),
)
COLUMNS = (
    "array folder system workload interface port_bits_in port_bits_out "
    "cycles_data_beats cycles_flow_reported lut ff dsp bram_18k_equiv wns_ns "
    "achieved_ns fmax_mhz unrouted scope"
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


def slack(text):
    found = re.search(r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)", text, re.S)
    return float(found.group(1)) if found else None


def routed(folder, size):
    util = read(folder, f"S{size}", "report", "util.rpt")
    row = {}
    if not util:
        return row
    row["lut"] = cell(util, "CLB LUTs")
    row["ff"] = cell(util, "CLB Registers")
    row["dsp"] = cell(util, "DSPs")
    tiles = cell(util, "Block RAM Tile")
    # 18K equivalents: a tile is one RAMB36 or two RAMB18, so twice the tiles.
    row["bram_18k_equiv"] = ("%g" % (2 * float(tiles))) if tiles else ""
    wns = slack(read(folder, f"S{size}", "report", "timing.rpt"))
    if wns is not None:
        row["wns_ns"] = "%.3f" % wns
        row["achieved_ns"] = "%.3f" % (TARGET_NS - wns)
        row["fmax_mhz"] = "%d" % round(1000.0 / (TARGET_NS - wns))
    status = read(folder, f"S{size}", "report", "route_status.rpt")
    found = re.search(r"# of nets with routing errors\.*\s*:\s*(\d+)", status)
    if found:
        row["unrouted"] = found.group(1)
    for log in ("build_pnr.log", "build.log"):
        found = re.search(
            r"(?:SPMW UNROUTED|PNR_UNROUTED) (\d+)",
            read(folder, f"S{size}", "report", log),
        )
        if found:
            row["unrouted"] = found.group(1)
    return row


def ports(folder, size):
    """Widest read master and the write master, as synthesised."""
    ins, outs = [], []
    for line in read(folder, f"S{size}", "report", "port_widths.txt").splitlines():
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


def cycles(folder, size, how):
    row = {}
    report = lambda name: read(folder, f"S{size}", "report", name)
    if how == "tokens":
        found = re.search(r"SPMW CYCLES total=(\d+)", report("cycles.txt"))
        ok = "SPMW COSIM PASS" in report("cycles.txt")
        if found and ok:
            row["cycles_data_beats"] = found.group(1)
    elif how == "spmw_tb":
        text = report("cycles.txt")
        found = re.search(r"first_beat (\d+) last_beat (\d+) cycles (\d+)", text)
        clean = re.search(r"RESULT 0 of \d+ byte", text)
        if found and clean:
            row["cycles_data_beats"] = found.group(3)
        start = re.search(r"starting at (\d+)", text)
        done = re.search(r"done after \d+ poll\(s\) at (\d+)", text)
        period = re.search(r"always #([\d.]+) ", report("tb.sv"))
        if start and done and period:
            ns = 2 * float(period.group(1))
            row["cycles_flow_reported"] = "%d" % round(
                (int(done.group(1)) - int(start.group(1))) / (1000 * ns)
            )
    else:
        found = re.search(
            r"BEATS first (\d+) last (\d+) cycles (\d+)", report("beats.txt")
        )
        verdict = report("cosim_verdict.txt")
        if found and "PASS" in verdict:
            row["cycles_data_beats"] = found.group(3)
        for name in ("kernel0_cosim.rpt", "gemm_cosim.rpt"):
            flow = re.search(
                r"\|\s*Verilog\|\s*Pass\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", report(name)
            )
            if flow:
                row["cycles_flow_reported"] = flow.group(3)
    return row


def rows():
    out = []
    for size in SIZES:
        for folder, name, scope, how in SYSTEMS:
            if not os.path.isdir(os.path.join(HERE, folder, f"S{size}")):
                continue
            row = dict.fromkeys(COLUMNS, "")
            row.update(
                array=f"{size}x{size}",
                folder=folder,
                system=name,
                workload=f"{size}^3",
                scope=scope,
            )
            row.update(routed(folder, size))
            row.update(cycles(folder, size, how))
            width = ports(folder, size)
            row.update(width)
            row["interface"] = (
                "streams"
                if scope == "array"
                else "512b" if width.get("port_bits_in") == "512" else "narrow"
            )
            out.append(row)
    return out


def num(text):
    return f"{int(float(text)):,}" if text else ""


def table(data):
    lines = [
        "| Array | Design | Port in/out | Cycles | (flow reports) | LUT | FF | DSP "
        "| BRAM18 | Slack | Clock |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    last = None
    for row in data:
        port = (
            "streams"
            if row["scope"] == "array"
            else (
                f"{row['port_bits_in']}/{row['port_bits_out']}"
                if row["port_bits_in"]
                else ""
            )
        )
        slack_ = f"{float(row['wns_ns']):+.3f} ns" if row["wns_ns"] else ""
        clock = f"{row['fmax_mhz']} MHz" if row["fmax_mhz"] else ""
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["array"] if row["array"] != last else "",
                row["system"],
                port,
                num(row["cycles_data_beats"]),
                num(row["cycles_flow_reported"]),
                num(row["lut"]),
                num(row["ff"]),
                num(row["dsp"]),
                row["bram_18k_equiv"],
                slack_,
                clock,
            )
        )
        last = row["array"]
    return "\n".join(lines)


def main():
    data = rows()
    with open(os.path.join(HERE, "results.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
    print(table(data))
    bad = [r for r in data if r["unrouted"] not in ("", "0")]
    if bad:
        sys.exit(
            "unrouted nets in: " + ", ".join(f"{r['folder']}/{r['array']}" for r in bad)
        )


if __name__ == "__main__":
    main()
