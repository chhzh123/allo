#!/usr/bin/env python3
"""Print LUT/FF/DSP/BRAM/WNS from a routed run directory. `read_pnr.py <dir>...`"""
import os
import re
import sys

for d in sys.argv[1:]:
    u = os.path.join(d, "util.rpt")
    t = os.path.join(d, "timing.rpt")
    if not os.path.isfile(u):
        print(f"{os.path.basename(d):34s} not routed")
        continue
    text = open(u, errors="replace").read()

    def cell(name, text=text):
        m = re.search(r"^\|\s*%s\*?\s*\|\s*([\d.]+)\s*\|" % re.escape(name), text, re.M)
        return m.group(1) if m else "-"

    wns = "-"
    if os.path.isfile(t):
        m = re.search(
            r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)",
            open(t, errors="replace").read(),
            re.S,
        )
        wns = m.group(1) if m else "-"
    print(
        f"{os.path.basename(d):34s} LUT {cell('CLB LUTs'):>7s}  FF {cell('CLB Registers'):>7s}  "
        f"DSP {cell('DSPs'):>4s}  BRAM {cell('Block RAM Tile'):>5s}  WNS {wns:>7s}"
    )
