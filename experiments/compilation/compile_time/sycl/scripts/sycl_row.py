# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One compile's stage times and what it built: python3 sycl_row.py <run dir>.

Prints `aoc_start_s,hls_done_s,kernels,cells,frac_dsps,aluts,ffs` from the
stamped build log and the early image's own summary report (its Total row)."""
import json
import os
import re
import sys

d = sys.argv[1]
log = open(os.path.join(d, "build.log"), errors="replace").read()


def stamp(text):
    found = re.search(r"^\s*([\d.]+) .*%s" % re.escape(text), log, re.M)
    return found.group(1) if found else ""


aoc = stamp("aoc: Processing SPIR-V")
hls = stamp("aoc: First stage compilation completed successfully")
kernels = cells = 0
dsps = aluts = ffs = 0.0
summary = os.path.join(d, "e1.prj", "reports", "resources", "json", "summary.ndjson")
if os.path.isfile(summary):
    names = {}
    for line in open(summary):
        row = json.loads(line)
        if row.get("parent") == "performanceSummary":
            kernels += 1
            cells += row.get("compiler_name", "").startswith("_ZTS4Cell")
        # The report's own total: kernels, interconnect and pipes.
        if row.get("parent") == "estimatedResources" and row.get("name") == "Total":
            aluts, ffs, dsps = (float(row["data"][i]) for i in (0, 1, 5))
print("%s,%s,%d,%d,%g,%d,%d" % (aoc, hls, kernels, cells, dsps, aluts, ffs))
