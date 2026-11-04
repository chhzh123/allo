#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_tapa_all.sh: E12's AutoSA 16x16 row (W = 64..512) through AutoSA's TAPA target.
# autosa_tapa_gen.sh must have run. Compiles one at a time (8 HLS jobs), then for every W
# the RTL simulation and the place and route side by side, each Vivado started
# only with 40 GB of memory free.
# Run from a directory that holds the autosa_tapa_*.sh drivers (E below).
main() {
set -u
local E=/scratch/hc676/e22_autosa_tapa/e12 W
cd $E
echo "ALL12_START $(date) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $E/driver.log
for W in 64 128 256 512; do
  bash autosa_tapa_compile.sh $W 8; echo "compile W=$W rc=$? $(date)" >> $E/driver.log
done
memfree() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
for W in 64 128 256 512; do
  [ -f $E/build/W$W/kernel0.xo ] || { echo "W=$W no .xo, skipped" >> $E/driver.log; continue; }
  while [ "$(memfree)" -lt 40 ]; do sleep 60; done
  (bash autosa_tapa_pnr.sh $W; echo "pnr W=$W done $(date)" >> $E/driver.log) &
  sleep 120
  while [ "$(memfree)" -lt 40 ]; do sleep 60; done
  (bash autosa_tapa_cosim.sh $W; echo "cosim W=$W done $(date)" >> $E/driver.log) &
  sleep 120
done
wait
echo "ALL12_DONE $(date)" >> $E/driver.log
}
main "$@"
