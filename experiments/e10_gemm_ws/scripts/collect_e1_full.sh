#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# collect_e1_full.sh: E1's SPMW kernel (`--design autosa`) routed with its
# movers and control (spmw_ws_kernel.sh <S> 4 autosa e1kernel), staged for
# experiments/e1_gemm/spmw_kernel/S<size>/report/full_kernel/.
set -u
E=/scratch/hc676/e15_ws
STAGE=$E/stage_e1
rm -rf $STAGE
for S in 4 8 16 32; do
  K=$E/spmw/e1kernel_S$S
  [ -f ${K}_pnr/util.rpt ] || continue
  D=$STAGE/S$S/report/full_kernel; mkdir -p $D
  cp ${K}_pnr/util.rpt ${K}_pnr/timing.rpt ${K}_pnr/util_hier.rpt ${K}_pnr/route_status.rpt ${K}_pnr/pnr.tcl $D/ 2>/dev/null
  grep -hE "SPMW TB" $K.sim.log > $D/cycles.txt 2>/dev/null
  cp $E/logs/spmw_e1kernel_S$S.log $D/build.log 2>/dev/null
  cp $K/spmw_kernel.sv $D/spmw_kernel.sv 2>/dev/null
done
find $STAGE -type f | wc -l
