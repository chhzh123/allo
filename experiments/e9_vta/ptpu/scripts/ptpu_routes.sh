#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# ptpu_routes.sh: the stream-fed engine's second route, at 3.0 ns, exported
# beside its first: each size's reports as p30_*, and both routes as one line
# each in routes.txt. The first route's line is the one already recorded.
set -u
E=/scratch/hc676/e12_ptpu
X=$E/export_p30/experiments/e9_vta/ptpu
rm -rf $E/export_p30; mkdir -p $X
for s in 4 8 16; do
  d=$E/route_${s}_p30
  [ -f $d/routed.dcp ] || continue
  R=$X/S$s/report; mkdir -p $R
  bash /scratch/hc676/e9_lean/hier.sh $d > $d/area_by_unit.txt 2>&1
  for f in util.rpt util_synth.rpt util_hier.rpt timing.rpt route.rpt area_by_unit.txt clock.xdc assemble.tcl; do
    [ -f $d/$f ] && cp $d/$f $R/p30_$f
  done
  grep -E "^(SPMW (STAGE|UNROUTED)|ARRAY WNS)" $d/vivado.log > $R/p30_pnr_stages.txt
  for b in b_ptpu-micro_${s}_r2 route_${s}_p30; do
    python3 - $E/$b $s <<'PY'
import re, sys
d, s = sys.argv[1], sys.argv[2]
target = float(open(d + "/clock.xdc").read().split()[2])
lines = open(d + "/timing.rpt", errors="replace").read().splitlines()
at = next(i for i, l in enumerate(lines) if "WNS(ns)" in l)
wns = float(lines[at + 2].split()[0])
try:
    hier = open(d + "/area_by_unit.txt").readline().split()
    lut, ff = hier[hier.index("LUT") + 1], hier[hier.index("FF") + 1]
except (OSError, ValueError):
    lut = ff = "?"
print(f"ROUTE size={s} target={target} wns={wns:+.3f} period={target - wns:.3f} lut={lut} ff={ff} dir={d.split('/')[-1]}")
PY
  done
done | tee $E/export_p30/routes_new.txt
