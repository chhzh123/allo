#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# routes.sh: every route of the whole engine as one line: its target, the
# clock it reached, and the engine's area on that route (`dut`, not the
# harness around it).
cd /scratch/hc676/e13_mem || exit 2
for s in 4 8 16; do
  for d in b_ptpumem-micro_$s $(ls -d route_${s}_p* 2>/dev/null); do
    [ -f $d/routed.dcp ] || continue
    [ -f $d/area_by_unit.txt ] || bash hier_mem.sh /scratch/hc676/e13_mem/$d > /dev/null 2>&1
    python3 - $d $s <<'PY'
import re, sys
d, s = sys.argv[1], sys.argv[2]
target = float(open(d + "/clock.xdc").read().split()[2])
lines = open(d + "/timing.rpt", errors="replace").read().splitlines()
at = next(i for i, l in enumerate(lines) if "WNS(ns)" in l)
wns = float(lines[at + 2].split()[0])
hier = open(d + "/area_by_unit.txt").readline().split()
get = lambda key: hier[hier.index(key) + 1]
worst = next(l for l in open(d + "/timing.rpt", errors="replace") if l.strip().startswith("Source:"))
unrouted = "0"
print(
    f"ROUTE size={s} target={target} wns={wns:+.3f} period={target - wns:.3f} "
    f"lut={get('LUT')} ff={get('FF')} ramb36={get('RAMB36')} ramb18={get('RAMB18')} "
    f"lutram={get('LUTRAM')} dir={d} worst={worst.split()[1]}"
)
PY
  done
done
