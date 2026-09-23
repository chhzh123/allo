#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# lean_hier.sh <build dir>: the routed array split into dut / cells / links / lanes.
# `--pnr` routes spmw_harness, which drives each edge channel from an LFSR; those
# are not the design, and Gemmini and VTA are routed bare, so `dut` is the row.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
B=$1
[ -f "$B/routed.dcp" ] || { echo "HIER NO_DCP $B"; exit 1; }
cd /scratch/hc676/e9_lean || exit 2
cat > "$B/hier.tcl" <<TCL
open_checkpoint $B/routed.dcp
report_utilization -hierarchical -hierarchical_depth 3 -file $B/util_hier.rpt
puts "HIER OK"
TCL
( cd "$B" && vivado -mode batch -source hier.tcl -nojournal -nolog > hier.log 2>&1 )
python3 - "$B" <<'PY'
import re, sys, collections
b = sys.argv[1]
rows = []
for line in open(b + "/util_hier.rpt", errors="replace"):
    m = re.match(r"^\|(\s*)(\S+)\s+\|\s*(\S+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", line)
    if m:
        depth = len(m.group(1)) // 2
        rows.append((depth, m.group(2), m.group(3), int(m.group(4)), int(m.group(8))))
# columns: Total LUTs | Logic LUTs | LUTRAMs | SRLs | FFs
hdr = [l for l in open(b + "/util_hier.rpt", errors="replace") if "Total LUTs" in l]
dut = [r for r in rows if r[1] == "dut"]
print("HIER", b.split("/")[-1], "dut LUT", dut[0][3], "FF", dut[0][4])
kinds = collections.defaultdict(lambda: [0, 0, 0])
for d, inst, mod, lut, ff in rows:
    if d != 2:
        continue
    key = re.sub(r"_\d+$", "", mod)
    key = re.sub(r"__parameterized\d+", "", key)
    kinds[key][0] += 1; kinds[key][1] += lut; kinds[key][2] += ff
for k, (n, lut, ff) in sorted(kinds.items(), key=lambda x: -x[1][1]):
    print(f"  {k:32s} n={n:4d} LUT={lut:7d} FF={ff:7d}  per={lut/n:7.1f}/{ff/n:7.1f}")
PY
