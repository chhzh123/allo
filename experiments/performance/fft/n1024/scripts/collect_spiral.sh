#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# collect_spiral.sh: stage the reports of the Spiral DFT generator's cores that
# the SwizzleFFT project simulated and routed (/scratch/hc676/spiral_runs, built
# by its experiments/spiral_point.sh), for E2 (N = 256) and E11 (N = 1,024).
# Copies reports only. The generated Verilog is "for internal, non-commercial
# research use only and is not for redistribution" and is left where it is.
set -u
RUNS=/scratch/hc676/spiral_runs
STAGE=/scratch/hc676/e14_fft/stage_spiral
rm -rf $STAGE
for N in 256 1024; do
  for W in 2 4 8 16; do
    R=$RUNS/n${N}_w$W
    [ -f $R/sim.json ] || continue
    D=$STAGE/N$N/W$W; mkdir -p $D
    cp $R/sim.json $R/sim.log $R/tb_spiral.v $D/
    cp $R/pnr.done $D/pnr.done
    for f in util.rpt timing.rpt route_status.rpt util_synth.rpt impl.tcl clock.xdc; do
      cp $R/pnr/$f $D/pnr_$f
    done
  done
done
cat /scratch/hc676/swizzle-fft/.version > $STAGE/swizzle_fft_version.txt 2>/dev/null
find $STAGE -type f | wc -l; du -sh $STAGE
