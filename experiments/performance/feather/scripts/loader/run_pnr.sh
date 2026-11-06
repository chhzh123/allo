#!/bin/bash
# OOC P&R of the row-wise-loader RTL at 4x4, 8x8, 16x16, against the corrected
# controller's already-recorded numbers.
set -u
W=/scratch/hc676/feather_loader
for N in 4 8 16; do
  echo "### pnr rowload N=$N  $(date +%T)"
  bash $W/work/pnr_ooc.sh $W/RTL_new $N $W/pnr/rowload_N$N
  cat $W/pnr/rowload_N$N/result.txt
  echo
done
echo "PNR ALL DONE $(date +%T)"
