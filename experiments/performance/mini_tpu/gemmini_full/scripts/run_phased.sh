#!/bin/bash
# run_phased.sh: the microbenchmark with every load issued before any compute,
# on both buses: the run in which the execute controller never waits.
cd /scratch/hc676/gemmini_full
M=/scratch/hc676/e9_lean/vta_bench/stim/stim_S16.txt
for d in 4 8 16; do
  PROGRAM=phased ./run_rocc_xsim.sh matmul $d $M microp &
  BUS=128 PROGRAM=phased ./run_rocc_xsim.sh matmul $d $M microp &
done
wait
echo PHASED_DONE
