#!/bin/bash
# attn16g.sh <G> <kind>...: attention's P.V on a 16x16 array cut into G column
# slabs, cosimulated at M = 1024, 2048, 4096 and 8192 query rows; <kind> is
# `grouped` (the whole reduction in one launch) or `conv` (one pass of the
# plain mesh, of which the product needs G). Every launch does 256
# multiply-adds a row whatever G is -- the reduction times the width is the PE
# count -- so G moves only the fill and the per-row rate.
#
#   ./attn16g.sh 8 grouped conv; ./attn16g.sh 2 grouped conv; ./attn16g.sh 4 grouped
#
# ALLO_TREE is the checkout to build from, ATTN16_OUT where the runs go; each
# run's log and `cost.json` are what `../results.csv` was made from.
main() {
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
local T=${ALLO_TREE:-/scratch/hc676/simtime/allo}
local E=${ATTN16_OUT:-/scratch/hc676/e20_attn16}
export PYTHONPATH=$T
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
local G=$1; shift
local kind seq D t0 rc
mkdir -p $E/logs
echo "ATTN16_G${G}_START $(date)" >> $E/logs/driver.log
for kind in "$@"; do
  for seq in 1024 2048 4096 8192; do
    D=$E/g${G}_${kind}_$seq
    rm -rf $D
    t0=$(date +%s)
    python3 $T/scripts/spmw_build_array.py --design attnpv16g${G}-$kind-$seq --size 16 \
      --frequency 300 --jobs 8 --cosim --out $D > $E/logs/g${G}_${kind}_$seq.log 2>&1
    rc=$?
    echo "G=$G $kind seq=$seq rc=$rc wall_s=$(( $(date +%s) - t0 )) $(grep -hoE 'SPMW COSIM [A-Z]+|SPMW CYCLES total=[0-9]+ first_out=[0-9]+' $E/logs/g${G}_${kind}_$seq.log | tr '\n' ' ')" >> $E/logs/driver.log
    rm -rf $D/sim/xsim.dir $D/ipgen $D/*/prj/sol/.autopilot
  done
done
echo "ATTN16_G${G}_DONE $(date)" >> $E/logs/driver.log
}
main "$@"
