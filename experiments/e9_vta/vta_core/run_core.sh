#!/bin/bash
# run_core.sh <width> <stim> <tag>: one stimulus on VTA's Core through its
# instruction stream (vta_core_bench.py).
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export PATH=/scratch/hc676/allo-agent/bin:$PATH
ROOT=/scratch/hc676/vta_build/core_bench
W=$1; STIM=$2; TAG=$3
RTL=/scratch/hc676/vta/hardware/chisel/vta_out_Core$([ "$W" = 16 ] || echo _w$W)
mkdir -p $ROOT
OUT=$ROOT/${TAG}_w$W
rm -rf "$OUT"
T0=$(date +%s)
python3 $ROOT/vta_core_bench.py --stim "$STIM" --out "$OUT" --width "$W" --rtl "$RTL" --tag "${TAG}_w$W" > $ROOT/${TAG}_w$W.log 2>&1
echo "rc=$? wall_s=$(( $(date +%s) - T0 ))" >> $ROOT/${TAG}_w$W.log
