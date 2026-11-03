# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""util.py <pnr dir>...: LUT, FF, DSP, BRAM18 (2 a 36K tile), WNS and the clock it
allows, 1000 / (3.333 - WNS) MHz -- the columns of experiments/e1_gemm/results.csv."""
import os, re, sys

for d in sys.argv[1:]:
    u = open(os.path.join(d, "util.rpt")).read()
    t = open(os.path.join(d, "timing.rpt")).read()
    num = lambda pat: float(
        re.search(r"\|\s*" + pat + r"\s*\|\s*([0-9.]+)", u).group(1)
    )
    wns = float(re.search(r"WNS\(ns\).*?\n.*?\n\s*(-?[0-9.]+)", t).group(1))
    print(
        "%-8s LUT %7d  FF %7d  DSP %5d  BRAM18 %4d  WNS %+.3f  %3.0f MHz"
        % (
            os.path.basename(d),
            num("CLB LUTs"),
            num("CLB Registers"),
            num("DSPs"),
            2 * num("Block RAM Tile"),
            wns,
            1000 / (3.333 - wns),
        )
    )
