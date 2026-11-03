#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pnr.sh <build>: E1's out-of-context Vivado flow (e1_autosa_wide/pnr/S*/pnr.tcl) on TAPA's RTL.
main() {
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
local B=$1 W=/scratch/hc676/e22_autosa_tapa
local R=$W/build/$B/work/hdl P=$W/pnr/$B
rm -rf $P; mkdir -p $P; cd $P
cat > pnr.tcl <<TCL
foreach f [glob -nocomplain $R/*.v] { read_verilog \$f }
foreach f [glob -nocomplain $R/*.sv] { read_verilog -sv \$f }
synth_design -top kernel0 -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 3.333 -name ap_clk [get_ports ap_clk]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file util.rpt
report_timing_summary -file timing.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
TCL
local t0=$(date +%s)
vivado -mode batch -source pnr.tcl > pnr.log 2>&1
echo "build=$B rc=$? wall_s=$(( $(date +%s) - t0 )) $(grep -o 'PNR_UNROUTED [0-9]*' pnr.log)" >> $W/pnr_results.txt
rm -rf .Xil
}
main "$@"
