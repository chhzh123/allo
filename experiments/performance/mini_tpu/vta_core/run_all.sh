#!/bin/bash
cd /scratch/hc676/vta_build/core_bench
for w in 4 8 16; do
  ./run_core.sh $w /scratch/hc676/e9_lean/vta_bench/stim/stim_S16.txt micro &
  ./run_core.sh $w /scratch/hc676/e10_llama/stim/llama_slice.txt llama &
  ./run_core.sh $w /scratch/hc676/e11_dsv4/stim/dsv4_slice.txt dsv4 &
done
wait
