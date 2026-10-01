#!/bin/bash
# run_all_sims.sh <variant>: the three workloads at the three sizes.
cd /scratch/hc676/gemmini_full
V=$1
for d in 4 8 16; do
  ./run_rocc_xsim.sh $V $d /scratch/hc676/e9_lean/vta_bench/stim/stim_S16.txt micro &
  ./run_rocc_xsim.sh $V $d /scratch/hc676/e10_llama/stim/llama_slice.txt llama &
  ./run_rocc_xsim.sh $V $d /scratch/hc676/e11_dsv4/stim/dsv4_slice.txt dsv4 &
done
wait
