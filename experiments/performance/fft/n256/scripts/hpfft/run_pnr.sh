#!/bin/bash
# usage: run_pnr.sh <size> <cfg> -- out-of-context Vivado synth+P&R at 3.333 ns of build/FFT_300MHz/syn/verilog
# (HLS-generated RTL plus the floating-point IP the RTL instantiates, created from HLS's own *_ip.tcl scripts).
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
E=/scratch/hc676/e2_hpfft
D=$E/$1/$2; S=$D/build/FFT_300MHz; V=$D/pnr
[ -d $S/syn/verilog ] || { echo "rc=99 no RTL in $S" > $D/pnr.done; exit 1; }
rm -rf $V; mkdir -p $V; cd $V
cp $S/syn/verilog/*.dat $V/ 2>/dev/null
echo "create_clock -period 3.333 -name ap_clk [get_ports ap_clk]" > clock.xdc
cat > impl.tcl <<EOF
set t0 [clock seconds]
create_project -force pnr $V/proj -part xcu280-fsvh2892-2L-e
set_property target_language Verilog [current_project]
add_files [glob $S/syn/verilog/*.v]
add_files -fileset constrs_1 $V/clock.xdc
foreach ipt [glob -nocomplain $S/syn/verilog/*_ip.tcl] { puts "E2 sourcing \$ipt"; source \$ipt }
set_property top FFT_TOP [current_fileset]
update_compile_order -fileset sources_1
set t1 [clock seconds]
synth_design -top FFT_TOP -part xcu280-fsvh2892-2L-e -mode out_of_context
set t2 [clock seconds]
report_utilization -file $V/util_synth.rpt
opt_design
set t3 [clock seconds]
place_design
set t4 [clock seconds]
phys_opt_design
set t5 [clock seconds]
route_design
set t6 [clock seconds]
report_utilization -file $V/util.rpt
report_utilization -hierarchical -hierarchical_depth 2 -file $V/util_hier.rpt
report_timing_summary -file $V/timing.rpt
report_route_status -file $V/route_status.rpt
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
puts "E2_PHASES setup=[expr \$t1-\$t0] synth=[expr \$t2-\$t1] opt=[expr \$t3-\$t2] place=[expr \$t4-\$t3] physopt=[expr \$t5-\$t4] route=[expr \$t6-\$t5]"
puts "E2 ROUTED WNS \$wns"
puts "E2 IMPLEMENTATION OK"
EOF
T0=$(date +%s)
vivado -mode batch -source impl.tcl -nojournal -log $V/vivado.log > $V/viv.stdout 2>&1
RC=$?
{ echo "rc=$RC total_wall_s=$(( $(date +%s) - T0 ))"; grep -E "E2_PHASES|E2 ROUTED WNS|E2 IMPLEMENTATION OK" $V/vivado.log; grep -E "^ERROR" $V/vivado.log | head -5; } > $D/pnr.done
