#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# all.sh: the E1 ladder through AutoSA's TAPA target, one step at a time so the
# compile timings do not share the machine with each other.
main() {
set -u
local W=/scratch/hc676/e22_autosa_tapa S
cd $W
echo "ALL_START $(date)" >> $W/driver.log
bash gen.sh 4 narrow >> $W/driver.log 2>&1
for S in 8 16 32; do bash gen.sh $S >> $W/driver.log 2>&1; done
for S in 4 8 16 32; do
  while [ "$(df -B1G --output=avail /scratch | tail -1)" -lt 20 ]; do sleep 60; done
  bash compile.sh $S 8; echo "compile S=$S j8 done $(date)" >> $W/driver.log
done
for S in 4 8 16 32; do bash cosim.sh $S S${S}_j8; echo "cosim S=$S done $(date)" >> $W/driver.log; done
for S in 4 8 16 32; do bash compile.sh $S 1; echo "compile S=$S j1 done $(date)" >> $W/driver.log; done
for S in 4 8 16 32; do bash pnr.sh S${S}_j8 & done; wait
echo "ALL_DONE $(date)" >> $W/driver.log
}
main "$@"
