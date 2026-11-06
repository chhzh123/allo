#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# hp_run.sh <size dir> <cfg>: one HP-FFT configuration end to end, on the
# recipe of E2's fixed variants: csim + csynth (run_hls.sh), the bounded
# cosimulation of 33 numpy transforms with a port-level VCD (run_cosim.sh),
# then an out-of-context route at 3.333 ns with the float IP generated.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
ulimit -n 8192
E=/scratch/hc676/e14_fft/hpfft
SZ=$1; CFG=$2
D=$E/$SZ/$CFG
LOG=$E/logs/${SZ}_$CFG.log
mkdir -p $E/logs
echo "HP_START $SZ $CFG $(date)" > $LOG
bash $E/run_hls.sh $SZ $CFG >> $LOG 2>&1
cat $D/hls.done >> $LOG
grep -q "rc=0" $D/hls.done || { echo "HP_FAIL hls" >> $LOG; exit 1; }
bash $E/run_cosim.sh $SZ $CFG >> $LOG 2>&1
cat $D/cosim.done >> $LOG
S=$D/build/FFT_300MHz/syn/verilog
OUT=$E/pnr/${SZ}_$CFG
rm -rf "$OUT"; mkdir -p "$OUT"
echo "create_clock -period 3.333 -name ap_clk [get_ports ap_clk]" > "$OUT/clock.xdc"
cat > "$OUT/impl.tcl" <<TCL
set t0 [clock seconds]
create_project -force pnr $OUT/proj -part xcu280-fsvh2892-2L-e
set_property target_language Verilog [current_project]
add_files [glob $S/*.v]
add_files -fileset constrs_1 $OUT/clock.xdc
foreach ipt [glob -nocomplain $S/*_ip.tcl] { source \$ipt }
set_property top FFT_TOP [current_fileset]
update_compile_order -fileset sources_1
set t1 [clock seconds]
synth_design -top FFT_TOP -part xcu280-fsvh2892-2L-e -mode out_of_context
set t2 [clock seconds]
report_utilization -file $OUT/util_synth.rpt
opt_design
place_design
phys_opt_design
route_design
set t3 [clock seconds]
report_utilization -file $OUT/util.rpt
report_utilization -hierarchical -hierarchical_depth 2 -file $OUT/util_hier.rpt
report_timing_summary -file $OUT/timing.rpt
report_route_status -file $OUT/route_status.rpt
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
puts "PNR_PHASES setup=[expr \$t1-\$t0] synth=[expr \$t2-\$t1] impl=[expr \$t3-\$t2]"
puts "PNR ROUTED WNS \$wns"
puts "PNR IMPLEMENTATION OK"
TCL
T0=$(date +%s)
( cd "$OUT" && vivado -mode batch -source impl.tcl -nojournal -log vivado.log > viv.stdout 2>&1 )
echo "rc=$? total_wall_s=$(( $(date +%s) - T0 ))" > $OUT.done
grep -E "^PNR" $OUT/vivado.log >> $OUT.done
cat $OUT.done >> $LOG
echo "HP_DONE $(date)" >> $LOG
