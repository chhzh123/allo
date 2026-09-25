#!/bin/bash
# pnr_acc.sh <dim>: Gemmini MxuAccVpu out of context, E3's recipe: 3.333 ns,
# xcu280, synth -> opt -> place -> phys_opt -> route, no retiming, no shell.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
D=$1
V=/scratch/hc676/e3_micro/gemmini_elab/mxuaccvpu_out_${D}
OUT=/scratch/hc676/e9_lean/gem/pnr_d${D}
rm -rf "$OUT"; mkdir -p "$OUT"
cat > "$OUT/pnr.tcl" <<TCL
read_verilog $V/MxuAccVpu.v
synth_design -top MxuAccVpu -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 3.333 -name clk [get_ports clock]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file util.rpt
report_utilization -hierarchical -hierarchical_depth 2 -file util_hier.rpt
report_timing_summary -file timing.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
TCL
T0=$(date +%s)
( cd "$OUT" && vivado -mode batch -source pnr.tcl -nojournal -nolog > pnr.log 2>&1 )
echo "PNR_D${D} rc=$? wall_s=$(( $(date +%s) - T0 ))" >> "$OUT/pnr.log"
