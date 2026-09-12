#!/bin/bash
# Split a routed SPMW array into the fabric and the harness that made it
# routable. `--pnr` routes `spmw_harness`, which wraps the array in one LFSR per
# channel so the stream ports stop being pins; those LFSRs are real logic and
# they are not the design. Gemmini's MxuVpu is routed bare, so the honest column
# to set beside it is the `dut` instance's, not the harness total.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1
OUT=/scratch/hc676/e3_micro/spmw_pnr_S${S}
cd /scratch/hc676 || exit 2                    # the driver stays outside the run dir
[ -f "$OUT/routed.dcp" ] || { echo "HIER_S${S} NO_DCP"; exit 1; }
cat > "$OUT/hier.tcl" <<TCL
open_checkpoint $OUT/routed.dcp
report_utilization -hierarchical -hierarchical_depth 2 -file $OUT/util_hier.rpt
puts "HIER OK"
TCL
( cd "$OUT" && vivado -mode batch -source hier.tcl -nojournal -nolog > hier.log 2>&1 )
echo "HIER_S${S} rc=$?"
grep -E "^\| +(spmw_harness|  dut)" "$OUT/util_hier.rpt" 2>/dev/null | head -4
