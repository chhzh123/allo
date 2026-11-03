#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# cmp2.sh: frontend + HLS compile time, AutoSA->TAPA against SPMW's mesh and kernel to
# RTL only (and the mesh as built, once). Phase A, 8 workers: one compile at a time,
# three repetitions, sizes 4..32, flows in a seeded random order per (rep, size).
# Phase B, 1 worker: every flow and size at once (each is one core). Runs already in
# runs.ndjson with rc 0 are kept, not repeated.
main() {
set -u
local C=/scratch/hc676/e22_autosa_tapa/cmp REP S order f
done_run() {
  python3 - "$C/runs.ndjson" "$1" "$2" "$3" "$4" <<'PY'
import json, sys
path, flow, size, jobs, rep = sys.argv[1:]
try:
    rows = [json.loads(l) for l in open(path)]
except OSError:
    rows = []
ok = any(r["flow"] == flow and r["size"] == int(size) and r["jobs"] == int(jobs)
         and r["rep"] == int(rep) and r["rc"] == 0 for r in rows)
sys.exit(0 if ok else 1)
PY
}
echo "CMP2_START $(date) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $C/driver.log
for REP in 0 1 2; do
  for S in 4 8 16 32; do
    local flows="tapa spmw_mesh_rtl spmw_kernel_rtl"
    [ $REP -eq 0 ] && flows="$flows spmw_mesh"
    order=$(python3 -c "
import random; r = random.Random(2203 + 100 * $REP + $S)
f = '$flows'.split(); r.shuffle(f); print(' '.join(f))")
    for f in $order; do
      done_run $f $S 8 $REP && continue
      while [ "$(df -B1G --output=avail /scratch | tail -1)" -lt 20 ]; do sleep 60; done
      bash $C/run_one.sh $f $S 8 $REP
      echo "done $f S=$S j=8 r=$REP $(date +%T) $(tail -1 $C/runs.ndjson | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print("rc", d["rc"], "total", d["total_s"])')" >> $C/driver.log
    done
  done
done
echo "PHASE_B_START $(date) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $C/driver.log
for S in 4 8 16 32; do
  for f in tapa spmw_mesh_rtl spmw_kernel_rtl spmw_mesh; do
    done_run $f $S 1 0 || bash $C/run_one.sh $f $S 1 0 &
  done
done
wait
echo "CMP2_DONE $(date)" >> $C/driver.log
}
main "$@"
