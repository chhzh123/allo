#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_tapa_compile.sh <W> <jobs>: tapa compile (analyze, per-task HLS, top-level RTL,
# .xo) of autosa_tapa_gen.sh's design for W^3, as E1's autosa_tapa/compile.sh.
main() {
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
local W=$1 J=$2 E=/scratch/hc676/e22_autosa_tapa/e12
local G=$E/gen/W$W/out/src D=$E/build/W$W
rm -rf $D; mkdir -p $D
local t0=$(date +%s.%N) l0=$(cut -d" " -f1-3 /proc/loadavg)
/usr/bin/time -v -o $D/time.txt tapa -w $D/work compile -f $G/kernel_kernel.cpp -t kernel0 \
  --part-num xcu280-fsvh2892-2L-e --clock-period 3.333 -j $J -o $D/kernel0.xo > $D/compile.log 2>&1
local rc=$? t1=$(date +%s.%N)
echo "W=$W jobs=$J rc=$rc wall_s=$(python3 -c "print(round($t1-$t0,1))") peak_kb=$(grep -oP 'Maximum resident set size \(kbytes\): \K[0-9]+' $D/time.txt) load_start=$l0 load_end=$(cut -d" " -f1-3 /proc/loadavg)" >> $E/compile_times.txt
return $rc
}
main "$@"
