#!/bin/bash
# usage: run_hls.sh <size> <cfg>  -- csim + csynth (project.tcl -> ../../common.tcl) in $E/<size>/<cfg>
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
ulimit -n 8192
E=/scratch/hc676/e2_hpfft
D=$E/$1/$2
cd $D || exit 1
rm -f hls.done
T0=$(date +%s)
timeout -k 120 21600 vitis_hls -f project.tcl > hls.log 2>&1
RC=$?
echo "rc=$RC hls_wall_s=$(( $(date +%s) - T0 )) start=$T0 end=$(date +%s)" > hls.done
