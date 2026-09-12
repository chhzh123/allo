#!/bin/bash
# OOC place-and-route of feather_top at 3.333 ns on xcu280, one size.
#   pnr_ooc.sh <RTL dir> <N> <out dir>
# Shipped defaults for everything but the array size (the authors own
# Figure-14 synthesis used the same: depth-4 SRAM reg arrays, DPE = N x N).
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
RTL=$1; N=$2; V=$3
mkdir -p $V; cd $V
echo "create_clock -period 3.333 -name clk [get_ports clk]" > clock.xdc
cat > impl.tcl <<EOT
create_project -in_memory -part xcu280-fsvh2892-2L-e
add_files [glob $RTL/*.v]
add_files -fileset constrs_1 $V/clock.xdc
proc stage {name body} {
  set t0 [clock milliseconds]
  uplevel 1 \$body
  puts "E4 STAGE \$name [expr {([clock milliseconds] - \$t0) / 1000.0}]"
}
stage synth { synth_design -top feather_top -part xcu280-fsvh2892-2L-e -mode out_of_context -generic DPE_COL_NUM=$N -generic DPE_ROW_NUM=$N }
report_utilization -file $V/util_synth.rpt
stage opt { opt_design }
stage place { place_design }
stage physopt { phys_opt_design }
stage route { route_design }
report_utilization -file $V/util.rpt
report_utilization -hierarchical -hierarchical_depth 1 -file $V/util_hier.rpt
report_timing_summary -file $V/timing.rpt
report_route_status -file $V/route.rpt
set wns [get_property SLACK [get_timing_paths -delay_type max]]
puts "E4 WNS \$wns"
set unrouted [llength [get_nets -quiet -filter {ROUTE_STATUS != ROUTED && ROUTE_STATUS != INTRASITE} -of [get_nets -quiet -hierarchical]]]
puts "E4 UNROUTED \$unrouted"
puts "IMPLEMENTATION OK"
EOT
T0=$(date +%s)
vivado -mode batch -source impl.tcl -nojournal -nolog > viv.log 2>&1
RC=$?
echo "rc=$RC seconds=$(( $(date +%s) - T0 ))" > result.txt
grep -E "^E4 |IMPLEMENTATION OK" viv.log >> result.txt
grep -iE "^ERROR" viv.log | head -5 >> result.txt
echo E4_PNR_DONE >> result.txt
