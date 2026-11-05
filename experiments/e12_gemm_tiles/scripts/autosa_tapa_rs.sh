#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_tapa_rs.sh <W> <jobs>: RapidStream's TAPA optimizer (floorplan + pipelined
# inter-slot FIFOs + export) on build/W<W>/kernel0.xo, whole-chip U280 model;
# its inputs come from autosa_tapa_rs_config.py.
# The shared July-2025 RapidStream: /opt/rapidstream (Jan 2025) fails its
# license check (KeyError: nonce) with the machine license.
main() {
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
local W=$1 J=$2 E=/scratch/hc676/e22_autosa_tapa/e12rs
local D=$E/rs/W$W
rm -rf $D; mkdir -p $D
local t0=$(date +%s)
/work/shared/common/rapidstream/rapidstream-tapaopt --work-dir $D --tapa-xo-path $E/build/W$W/kernel0.xo \
  --device-config $E/u280_default.json --floorplan-config $E/floorplan_config.json \
  --pipeline-config $E/pipeline_config.json -j $J > $D/../rs_W$W.log 2>&1
local rc=$?
echo "W=$W rc=$rc wall_s=$(( $(date +%s) - t0 ))" >> $E/rs_results.txt
return $rc
}
main "$@"
