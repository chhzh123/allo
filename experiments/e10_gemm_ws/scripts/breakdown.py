#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where each kernel's area is, from its routed hierarchical utilisation.

    python3 scripts/breakdown.py [size ...]      # default: every size present

Three parts, the same three for every system, so the split can be compared:

    array    the cells, the links between them and the buffers at their edge
    network  what carries operands and results between the memory masters and
             the array: AutoSA's I/O modules and their FIFOs, Allo's load and
             store loops and tile buffers, the body of an SPMW mover
    memory   the AXI masters themselves and the control interface

The reports are `report_utilization -hierarchical` of the routed design
(`report/util_hier.rpt`), one or two levels deep.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ("Total LUTs", "FFs", "RAMB36", "RAMB18", "DSP Blocks")


def load(folder, size):
    path = os.path.join(HERE, folder, f"S{size}", "report", "util_hier.rpt")
    if not os.path.isfile(path):
        return []
    rows, header = [], None
    with open(path, errors="replace") as handle:
        for line in handle:
            if not line.startswith("|"):
                continue
            raw = line.rstrip("\n").strip("|").split("|")
            cells = [c.strip() for c in raw]
            if cells[0] == "Instance":
                header = cells
                continue
            if header is None or len(cells) != len(header):
                continue
            depth = (len(raw[0]) - len(raw[0].lstrip())) // 2
            row = dict(zip(header, cells))
            rows.append((depth, row["Instance"], [int(row[k]) for k in KEYS]))
    return rows


def add(total, values):
    for k, value in enumerate(values):
        total[k] += value


def spmw(rows):
    """top > dut | u_control | u_feed_* | u_drain_* > gmem_m_axi_U | loop."""
    parts = {name: [0] * 5 for name in ("array", "network", "memory")}
    mover = None
    for depth, name, values in rows[1:]:
        if depth == 1:
            mover = None
            if name == "dut":
                add(parts["array"], values)
            elif name.startswith("u_control"):
                add(parts["memory"], values)
            else:
                mover = values
                add(parts["network"], values)
        elif depth == 2 and mover is not None and "m_axi" in name:
            add(parts["memory"], values)
            add(parts["network"], [-v for v in values])
    return parts


# A cell of AutoSA's array is a PE, the module that holds its weight
# (B_IO_L1_in) and the FIFOs between the two and between neighbours.
AUTOSA = (
    (
        "array",
        r"^(PE_wrapper|B_IO_L1_in|fifo_(A|B|C_1)_PE|fifo_B_B_IO_L1_in"
        r"|start_for_(PE|B_IO_L1_in))",
    ),
    ("memory", r"^(gmem_\w+_m_axi|control_s_axi)"),
    ("network", r"."),
)


def autosa(rows):
    parts = {name: [0] * 5 for name in ("array", "network", "memory")}
    for depth, name, values in rows[1:]:
        if depth != 1:
            continue
        for part, pattern in AUTOSA:
            if re.search(pattern, name):
                add(parts[part], values)
                break
    return parts


def allo(rows):
    """top > gmem*_m_axi_U | grp_systolic_ws_gemm > grp_systolic_tile_ws_gemm."""
    parts = {name: [0] * 5 for name in ("array", "network", "memory")}
    for depth, name, values in rows[1:]:
        if depth == 1 and re.match(r"gmem\d_m_axi|control", name):
            add(parts["memory"], values)
        elif depth == 2 and "systolic_tile_ws" in name:
            add(parts["array"], values)
    top = rows[0][2]
    parts["network"] = [
        t - a - m for t, a, m in zip(top, parts["array"], parts["memory"])
    ]
    return parts


SYSTEMS = (
    ("spmw_kernel", "SPMW kernel", spmw),
    ("autosa", "AutoSA", autosa),
    ("autosa_wide", "AutoSA, wide", autosa),
    ("allo", "Allo", allo),
)


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [4, 8, 16, 32]
    print("| Array | Design | part | LUT | FF | BRAM18 | DSP |")
    print("|---|---|---|---:|---:|---:|---:|")
    for size in sizes:
        first = True
        for folder, label, split in SYSTEMS:
            rows = load(folder, size)
            if not rows:
                continue
            parts = split(rows)
            for k, part in enumerate(("array", "network", "memory")):
                lut, ff, b36, b18, dsp = parts[part]
                print(
                    "| %s | %s | %s | %s | %s | %d | %d |"
                    % (
                        f"{size}x{size}" if first else "",
                        label if k == 0 else "",
                        part,
                        f"{lut:,}",
                        f"{ff:,}",
                        2 * b36 + b18,
                        dsp,
                    )
                )
                first = False


if __name__ == "__main__":
    main()
