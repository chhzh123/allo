#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_tapa_rs_cosim.sh <W>: autosa_tapa_cosim.sh (same host, same beat counter) on RapidStream's
# updated.xo, so the cycles include the pipeline registers it added. At 64^3 its
# elaboration reached 115 GB and ran brg-zhang-xcel out of memory: not rerun.
main() {
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
local W=$1 E=/scratch/hc676/e22_autosa_tapa/e12rs
local H=/scratch/hc676/e22_autosa_tapa/e12/gen/W$W/out/src/host X=$E/rs/W$W/dse/solution_0/updated.xo C=$E/cosim/W$W
rm -rf $C; mkdir -p $C
local t0=$(date +%s)
TAPA_BITSTREAM=$X $H --xosim_work_dir=$C/work --xosim_part_num=xcu280-fsvh2892-2L-e \
  --xosim_executable=/scratch/hc676/e22_autosa_tapa/fastcosim_beats.sh > $C/cosim.log 2>&1
local rc=$? t1=$(date +%s)
echo "W=$W rc=$rc $(grep -h -o "BEATS.*" $C/cosim.log | head -1) check=$(grep -h -o "Passed!\|Failed with [0-9]* errors" $C/cosim.log | head -1) wall_s=$((t1-t0))" >> $E/cosim_results.txt
}
main "$@"
