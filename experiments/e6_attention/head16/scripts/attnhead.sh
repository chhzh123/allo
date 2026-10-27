#!/bin/bash
# attnhead.sh <S>...: one attention head's whole P.V, (S, S) @ (S, 8), on a
# 16x16 array in one launch, cosimulated for both schemes: `grouped` (G = 2,
# 32 keys a tile) and `conv` (the plain mesh, 16 keys a tile, 8 columns empty).
#
#   ./attnhead.sh 64 128 256 512 1024
#
# ALLO_TREE is the checkout to build from, HEAD16_OUT where the runs go; each
# run's log and `cost.json` are what `../results.csv` was made from.
main() {
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
local T=${ALLO_TREE:-/scratch/hc676/simtime/allo}
local E=${HEAD16_OUT:-/scratch/hc676/e21_attnhead}
export PYTHONPATH=$T
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
local seq kind D t0 rc attempt
mkdir -p $E/logs
echo "HEAD16_START $(date)" >> $E/logs/driver.log
for seq in "$@"; do
  for kind in grouped conv; do
    D=$E/${kind}_$seq
    t0=$(date +%s)
    # /scratch is shared and was full once: a unit's HLS fails with an
    # internal error when it cannot write, so wait for room and retry once.
    for attempt in 1 2; do
      while [ "$(df -B1G --output=avail /scratch | tail -1)" -lt 2 ]; do sleep 60; done
      rm -rf $D
      python3 -u $T/scripts/spmw_build_array.py --design attnhead16d8-$kind-$seq --size 16 \
        --frequency 300 --jobs 8 --cosim --out $D > $E/logs/${kind}_$seq.log 2>&1
      rc=$?
      [ $rc -eq 0 ] && break
    done
    echo "$kind S=$seq rc=$rc wall_s=$(( $(date +%s) - t0 )) $(grep -hoE 'SPMW COSIM [A-Z]+|SPMW CYCLES total=[0-9-]+ first_out=[0-9]+ first_in=[0-9]+' $E/logs/${kind}_$seq.log | tr '\n' ' ')" >> $E/logs/driver.log
    rm -rf $D/sim/xsim.dir $D/ipgen $D/*/prj/sol/.autopilot
  done
done
echo "HEAD16_DONE $(date)" >> $E/logs/driver.log
}
main "$@"
