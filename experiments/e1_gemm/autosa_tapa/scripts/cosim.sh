#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# cosim.sh <S> <build>: TAPA fast cosim of build/<build>/kernel0.xo with the beat
# counter of fastcosim_beats.sh; the host checks C against its own golden model.
main() {
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
local S=$1 B=$2 W=/scratch/hc676/e22_autosa_tapa
local H=$W/gen/S$S/out/src/host X=$W/build/$B/kernel0.xo C=$W/cosim/$B
rm -rf $C; mkdir -p $C
local t0=$(date +%s)
TAPA_BITSTREAM=$X $H --xosim_work_dir=$C/work --xosim_part_num=xcu280-fsvh2892-2L-e \
  --xosim_executable=$W/fastcosim_beats.sh > $C/cosim.log 2>&1
local rc=$? t1=$(date +%s)
echo "S=$S build=$B rc=$rc $(grep -h -o "BEATS.*" $C/cosim.log | head -1) check=$(grep -h -o "Passed!\|Failed with [0-9]* errors" $C/cosim.log | head -1) wall_s=$((t1-t0))" >> $W/cosim_results.txt
}
main "$@"
