#!/bin/bash
# scope_timing.sh <pnr dir>...: time each routed Gemmini in the execute scope.
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
ROOT=/scratch/hc676/gemmini_full
export TMPDIR=$ROOT/tmp; mkdir -p "$TMPDIR"
for d in "$@"; do
  ( cd "$ROOT/$d" && vivado -mode batch -source $ROOT/gemmini_scope_timing.tcl -nojournal -nolog > scope.log 2>&1
    grep -E "^(SCOPE|EXECUTE|MESH|WHOLE)_" scope.log > scope.txt; echo "rc=$?" >> scope.txt )
done
