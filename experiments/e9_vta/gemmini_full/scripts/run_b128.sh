#!/bin/bash
# run_b128.sh: the matmul-only Gemmini on Chipyard-width (128-bit) system bus,
# the three workloads at the three sizes, library and hand-scheduled programs.
cd /scratch/hc676/gemmini_full
for d in 8 16; do BUS=128 SIM=1 ./elab.sh $d matmul; done
M=/scratch/hc676/e9_lean/vta_bench/stim/stim_S16.txt
for d in 4 8 16; do
  BUS=128 ./run_rocc_xsim.sh matmul $d $M micro &
  BUS=128 PROGRAM=batched ./run_rocc_xsim.sh matmul $d $M microb &
  BUS=128 ./run_rocc_xsim.sh matmul $d /scratch/hc676/e10_llama/stim/llama_slice.txt llama &
  BUS=128 ./run_rocc_xsim.sh matmul $d /scratch/hc676/e11_dsv4/stim/dsv4_slice.txt dsv4 &
done
wait
echo B128_DONE
