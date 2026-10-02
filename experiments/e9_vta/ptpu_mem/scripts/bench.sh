#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# bench.sh <workload> <size> [build]: the memory bench of one workload on one
# size's build. STALL=<percent> withholds the memory's handshakes; TAG names
# the run's directory.
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export PYTHONPATH=/scratch/hc676/allo:/scratch/hc676/allo/tests/dataflow/spmw
W=$1; S=$2
BUILD=${3:-/scratch/hc676/e13_mem/b_ptpumem-micro_$S}
case $W in
  micro) WHAT="--stim /scratch/hc676/e9_lean/vta_bench/stim/stim_S16.txt" ;;
  llama) WHAT="--stim /scratch/hc676/e10_llama/stim/llama_slice.txt" ;;
  dsv4)  WHAT="--stim /scratch/hc676/e11_dsv4/stim/dsv4_slice.txt" ;;
  mixed) WHAT="--mixed" ;;
  *) echo "unknown workload $W"; exit 2 ;;
esac
OUT=/scratch/hc676/e13_mem/bench/${W}_$S${TAG:-}
rm -rf "$OUT"; mkdir -p "$OUT"
cd /scratch/hc676/allo || exit 2
python3 -u experiments/e9_vta/scripts/spmw_mem_bench.py $WHAT --build "$BUILD" --out "$OUT" --size "$S" --stall "${STALL:-0}" --tag "${W}_$S${TAG:-}" > "$OUT/bench.log" 2>&1
echo "BENCH_RC=$?" >> "$OUT/bench.log"
