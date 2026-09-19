#!/bin/bash
# One VTA module, placed and routed out of context on the U280 -- the same
# recipe as Gemmini's `pnr_norm.sh`, retiming included, so the two are
# comparable.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export TMPDIR=/scratch/hc676/vta_build/tmp; mkdir -p "$TMPDIR"
TOP=$1
PERIOD=${2:-3.333}
V=/scratch/hc676/vta/hardware/chisel/vta_out_$TOP
OUT=/scratch/hc676/vta_build/pnr_${TOP}_${PERIOD/./p}
cd /scratch/hc676 || exit 2
[ -f "$V/$TOP.v" ] || { echo "NO_RTL in $V"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
cat > "$OUT/pnr.tcl" <<TCL
foreach f [glob -nocomplain $V/*.v]  { read_verilog \$f }
foreach f [glob -nocomplain $V/*.sv] { read_verilog -sv \$f }
synth_design -top $TOP -part xcu280-fsvh2892-2L-e -mode out_of_context -retiming
create_clock -period $PERIOD -name clk [get_ports clock]
opt_design
place_design
phys_opt_design -retime
route_design
phys_opt_design -retime
report_utilization -file util.rpt
report_timing_summary -file timing.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
TCL
T0=$(date +%s)
( cd "$OUT" && exec vivado -mode batch -source pnr.tcl -nojournal -nolog > pnr.log 2>&1 )
echo "rc=$? wall_s=$(( $(date +%s) - T0 ))" > "$OUT/DONE"
grep -E "^PNR_UNROUTED" "$OUT/pnr.log" >> "$OUT/DONE" 2>/dev/null
