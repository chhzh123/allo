#!/bin/bash
# FEATHER's SPMW port routed with its integer multiply bound to fabric.
#
# The published RTL routes with **zero** DSP blocks at every size -- a plain
# `*` in Verilog that Vivado maps to lookup tables -- while the port's
# identical multiply was inferred into one DSP per element: 16, 64, 256. Two
# lookup-table counts only compare if both designs put the multiplier in the
# same place, so this routes the port the RTL's way. It is expected to cost
# LUTs; that is the honest number rather than a smaller one bought by moving
# arithmetic off the fabric being counted.
#
#   run_fabmul_pnr.sh <size>
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
export PYTHONDONTWRITEBYTECODE=1
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
N=$1
OUT=/scratch/hc676/e4_pnr/spmw_feather_fabmul_$N
cd /scratch/hc676/allo || exit 2          # the driver stays outside the run dir
rm -rf "$OUT"; mkdir -p "$OUT"
T0=$(date +%s)
SPMW_BIND_MUL=1 python3 scripts/spmw_build_array.py --design feather --size "$N" \
  --frequency 300 --jobs 6 --pnr --out "$OUT" > "$OUT/build.log" 2>&1
echo "rc=$? seconds=$(( $(date +%s) - T0 ))" > "$OUT/E4_DONE"
