#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_recosim.sh <S> <W> <narrow|wide>: run autosa_tiles.sh's cosimulation
# again on the design it already synthesised, with the stack its host program
# needs. For the 1024-cubed build whose first cosimulation ended in the C
# testbench, before autosa_tiles.sh raised the limit itself.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1; W=$2; V=${3:-narrow}
E=/scratch/hc676/e16_tiles
D=$E/autosa/$V/S${S}_W$W; H=$D/hls
LOG=$E/logs/autosa_${V}_S${S}_W$W.log
cd $H || exit 1
cat > recosim.tcl <<TCL
open_project cosim_prj
open_solution sol
cosim_design -rtl verilog
exit
TCL
ulimit -s unlimited 2>/dev/null || ulimit -s 1048576
echo "recosim: stack $(ulimit -s) $(date)" >> $LOG
cp -p cosim.log cosim_first.log
T0=$(date +%s)
vitis_hls -f recosim.tcl > cosim.log 2>&1
echo "recosim hls rc=$? wall_s=$(( $(date +%s) - T0 ))" >> $LOG
grep -E "^Passed!|^Failed with|C/RTL co-simulation finished|DEADLOCK DETECTED" $H/cosim.log | head -4 >> $LOG
grep -E "\| *kernel0 *\||Verilog" $H/cosim_prj/sol/sim/report/*_cosim.rpt 2>/dev/null | head -3 | cut -c1-140 >> $LOG
echo "AUTOSA_RECOSIM_DONE $(date)" >> $LOG
