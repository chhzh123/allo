#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# spmw_ws.sh <S> [jobs]: SPMW's weight-stationary S x S array on one S x S x S
# matmul, the array alone (E1's e1_spmw.sh): cosimulation, whose cycles run
# from the first input token to the last output token, and place and route at
# 300 MHz with its boundary buffers. spmw_ws_kernel.sh does the kernel.
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export PYTHONPATH=/scratch/hc676/allo:/scratch/hc676/allo/tests/dataflow/spmw
S=$1; J=${2:-4}
E=/scratch/hc676/e15_ws; W=$E/spmw; mkdir -p $W $E/logs
LOG=$E/logs/spmw_mesh_S$S.log
cd /scratch/hc676/allo
echo "SPMW_WS_MESH_START S=$S $(date)" > $LOG
for mode in cosim pnr; do
  D=$W/mesh_S${S}_$mode; rm -rf $D
  T0=$(date +%s)
  python3 -u scripts/spmw_build_array.py --design gemmws --size $S --frequency 300 \
    --jobs $J --$mode --out $D > $D.log 2>&1
  echo "$mode rc=$? wall_s=$(( $(date +%s) - T0 ))" >> $LOG
  grep -hE "SPMW COSIM|SPMW CYCLES|array clock|SPMW UNROUTED|^gemmws:|^HLS:|Vivado implemented" $D.log \
    | cut -c1-200 >> $LOG
done
echo "SPMW_WS_MESH_DONE $(date)" >> $LOG
