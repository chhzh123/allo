#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# compile.sh <S> <jobs>: tapa compile of the AutoSA TAPA output for an SxS array.
main() {
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
local S=$1 J=$2 W=/scratch/hc676/e22_autosa_tapa
local G=$W/gen/S$S/out/src D=$W/build/S${S}_j$J
rm -rf $D; mkdir -p $D
local t0=$(date +%s.%N)
/usr/bin/time -v -o $D/time.txt tapa -w $D/work compile -f $G/kernel_kernel.cpp -t kernel0 \
  --part-num xcu280-fsvh2892-2L-e --clock-period 3.333 -j $J --keep-hls-work-dir \
  -o $D/kernel0.xo > $D/compile.log 2>&1
local rc=$?
local t1=$(date +%s.%N)
echo "S=$S jobs=$J rc=$rc wall_s=$(python3 -c "print(round($t1-$t0,1))") peak_kb=$(grep -oP 'Maximum resident set size \(kbytes\): \K[0-9]+' $D/time.txt)" >> $W/compile_times.txt
}
main "$@"
