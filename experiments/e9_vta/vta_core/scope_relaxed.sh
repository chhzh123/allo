#!/bin/bash
# scope_relaxed.sh: the datapath + scratchpads scope, timed on the relaxed-target routes.
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export TMPDIR=/scratch/hc676/vta_build/tmp
for d in pnr_Core_w4_p43 pnr_Core_w4_p46 pnr_Core_w8_p43 pnr_Core_w8_p46 pnr_Core_w16_p43 pnr_Core_w16_p46; do
  ( cd /scratch/hc676/vta_build/$d && vivado -mode batch -source ../scope_timing.tcl -nojournal -nolog > scope.log 2>&1 ) &
done
wait
echo SCOPE_RELAXED_DONE
