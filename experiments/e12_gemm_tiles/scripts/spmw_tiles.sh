#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# spmw_tiles.sh <S> <W> <cosim|pnr> [jobs]: SPMW's S x S output-stationary
# array on a W x W x W matmul, the array alone. `cosim` simulates the
# assembled RTL on the whole workload against numpy; its cycles run from the
# first input token to the last output token. `pnr` places and routes the
# array at 300 MHz with its boundary buffers; the hardware does not depend on
# W, so one route a size is enough. spmw_tiles_kernel.sh does the kernel.
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export PYTHONPATH=/scratch/hc676/allo:/scratch/hc676/allo/tests/dataflow/spmw
S=$1; W=$2; MODE=$3; J=${4:-4}
E=/scratch/hc676/e16_tiles; O=$E/spmw; mkdir -p $O $E/logs
LOG=$E/logs/spmw_mesh_S${S}_W${W}_$MODE.log
D=$O/mesh_S${S}_W${W}_$MODE; rm -rf $D
cd /scratch/hc676/allo
echo "SPMW_TILES_MESH_START S=$S W=$W $MODE $(date)" > $LOG
T0=$(date +%s)
python3 -u scripts/spmw_build_array.py --design gemmtiles-$W --size $S --frequency 300 \
  --jobs $J --$MODE --out $D > $D.log 2>&1
echo "$MODE rc=$? wall_s=$(( $(date +%s) - T0 ))" >> $LOG
grep -hE "SPMW COSIM|SPMW CYCLES|array clock|SPMW UNROUTED|^gemmtiles|^HLS:|Vivado implemented|was asked" $D.log \
  | cut -c1-200 >> $LOG
echo "SPMW_TILES_MESH_DONE $(date)" >> $LOG
