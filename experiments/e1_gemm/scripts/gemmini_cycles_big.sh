#!/bin/bash
# Gemmini weight-stationary mesh cycles at 16x16 and 32x32. The earlier sweep
# ran all four sizes under `timeout 5400`, and 16 and 32 hit that ceiling and
# produced nothing -- which is why those cells are blank, not because the
# driver failed. Same driver, same checks, six hours each, run one at a time
# because sbt elaboration of a 32x32 mesh is memory-hungry.
set -u
cd /scratch/hc676/gemmini_build || exit 2
source /scratch/hc676/gemmini_env.sh
for S in 16 32; do
  echo "GEMMINI_BIG_S${S} START $(date -Is)"
  T0=$(date +%s)
  timeout 21600 env MESH_DIM=$S sbt -batch "runMain gen.MeshDriver" > /scratch/hc676/gemmini_S$S.log 2>&1
  rc=$?
  echo "GEMMINI_BIG_S${S} rc=$rc wall_s=$(( $(date +%s) - T0 ))"
  if grep -qE "GEMMINI_CYCLES" /scratch/hc676/gemmini_S$S.log; then
    grep -E "GEMMINI_CYCLES" /scratch/hc676/gemmini_S$S.log | head -2
  else
    echo "GEMMINI_BIG_S${S} NO_CYCLES rc=$rc  124 means it hit the six-hour cap"
    grep -E "^\[error\]|Exception|OutOfMemory" /scratch/hc676/gemmini_S$S.log | head -5
  fi
  echo "GEMMINI_BIG_S${S} END $(date -Is)"
done
echo GEMMINI_BIG_DONE
