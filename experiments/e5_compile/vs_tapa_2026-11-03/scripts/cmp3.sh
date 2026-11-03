#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# cmp3.sh: rerun AutoSA->TAPA with `tapa ... link` (its complete top-level RTL) and in
# its own environment -- 8 workers, three repetitions, one compile at a time -- then
# every flow with 1 worker at once, as cmp2.sh's phase B. Earlier TAPA rows and every
# 1-worker row were removed from runs.ndjson (kept in runs_v2.ndjson).
main() {
set -u
local C=/scratch/hc676/e22_autosa_tapa/cmp REP S f
echo "CMP3_START $(date) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $C/driver.log
for REP in 0 1 2; do
  for S in 4 8 16 32; do
    bash $C/run_one.sh tapa $S 8 $REP
    echo "done tapa S=$S j=8 r=$REP $(date +%T) $(tail -1 $C/runs.ndjson | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print("rc", d["rc"], "total", d["total_s"])')" >> $C/driver.log
  done
done
echo "PHASE_B_START $(date) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $C/driver.log
for S in 4 8 16 32; do
  for f in tapa spmw_mesh_rtl spmw_kernel_rtl spmw_mesh; do
    bash $C/run_one.sh $f $S 1 0 &
  done
done
wait
echo "CMP3_DONE $(date)" >> $C/driver.log
}
main "$@"
