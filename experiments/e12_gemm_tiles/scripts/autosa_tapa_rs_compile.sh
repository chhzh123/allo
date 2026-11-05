#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_tapa_rs_compile.sh <W> <jobs>: autosa_tapa_compile.sh's tapa compile (same
# AutoSA output, same TAPA 0.1.20250127) into the RapidStream tree, e12rs.
main() {
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
local W=$1 J=$2 E=/scratch/hc676/e22_autosa_tapa/e12rs
local G=/scratch/hc676/e22_autosa_tapa/e12/gen/W$W/out/src D=$E/build/W$W
rm -rf $D; mkdir -p $D
local t0=$(date +%s.%N)
tapa -w $D/work compile -f $G/kernel_kernel.cpp -t kernel0 \
  --part-num xcu280-fsvh2892-2L-e --clock-period 3.333 -j $J -o $D/kernel0.xo > $D/compile.log 2>&1
local rc=$? t1=$(date +%s.%N)
echo "W=$W jobs=$J rc=$rc wall_s=$(python3 -c "print(round($t1-$t0,1))")" >> $E/compile_times.txt
return $rc
}
main "$@"
