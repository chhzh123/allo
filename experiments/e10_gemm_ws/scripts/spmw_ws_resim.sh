#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# spmw_ws_resim.sh <S> [jobs]: simulate the built array and kernel again after
# the design was given operands that span int8 (`spmw_operands`). The hardware
# does not depend on the operands, so the kernel package and both routes
# stand; only the two simulations are rerun.
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export PYTHONPATH=/scratch/hc676/allo:/scratch/hc676/allo/tests/dataflow/spmw
S=$1; J=${2:-4}
E=/scratch/hc676/e15_ws; W=$E/spmw
cd /scratch/hc676/allo
K=$W/kernel_S$S
LOG=$E/logs/spmw_kernel_S$S.log
echo "RESIM int8-spanning operands $(date)" >> $LOG
rm -rf ${K}_sim ${K}_ops
timeout 1200 python3 -u scripts/spmw_array_operands.py --design gemmws --size $S \
  --out ${K}_ops > $K.ops.log 2>&1
echo "operands rc=$?" >> $LOG
timeout 7200 python3 -u scripts/spmw_kernel_sim.py $K ${K}_ops --design gemmws \
  --size $S > $K.sim.log 2>&1
echo "sim rc=$?" >> $LOG
grep -hE "SPMW TB|xsim:" $K.sim.log | cut -c1-200 >> $LOG
LOG=$E/logs/spmw_mesh_S$S.log
echo "RESIM int8-spanning operands $(date)" >> $LOG
D=$W/mesh_S${S}_cosim; rm -rf $D
python3 -u scripts/spmw_build_array.py --design gemmws --size $S --frequency 300 \
  --jobs $J --cosim --out $D > $D.log 2>&1
echo "cosim rc=$?" >> $LOG
grep -hE "SPMW COSIM|SPMW CYCLES" $D.log | cut -c1-200 >> $LOG
echo "SPMW_WS_RESIM_DONE $(date)" >> $LOG
