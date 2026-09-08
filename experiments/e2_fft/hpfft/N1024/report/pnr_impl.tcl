set t0 [clock seconds]
create_project -force pnr /scratch/hc676/e2_hpfft/n1024/UF1/pnr/proj -part xcu280-fsvh2892-2L-e
set_property target_language Verilog [current_project]
add_files [glob /scratch/hc676/e2_hpfft/n1024/UF1/build/FFT_300MHz/syn/verilog/*.v]
add_files -fileset constrs_1 /scratch/hc676/e2_hpfft/n1024/UF1/pnr/clock.xdc
foreach ipt [glob -nocomplain /scratch/hc676/e2_hpfft/n1024/UF1/build/FFT_300MHz/syn/verilog/*_ip.tcl] { puts "E2 sourcing $ipt"; source $ipt }
set_property top FFT_TOP [current_fileset]
update_compile_order -fileset sources_1
set t1 [clock seconds]
synth_design -top FFT_TOP -part xcu280-fsvh2892-2L-e -mode out_of_context
set t2 [clock seconds]
report_utilization -file /scratch/hc676/e2_hpfft/n1024/UF1/pnr/util_synth.rpt
opt_design
set t3 [clock seconds]
place_design
set t4 [clock seconds]
phys_opt_design
set t5 [clock seconds]
route_design
set t6 [clock seconds]
report_utilization -file /scratch/hc676/e2_hpfft/n1024/UF1/pnr/util.rpt
report_utilization -hierarchical -hierarchical_depth 2 -file /scratch/hc676/e2_hpfft/n1024/UF1/pnr/util_hier.rpt
report_timing_summary -file /scratch/hc676/e2_hpfft/n1024/UF1/pnr/timing.rpt
report_route_status -file /scratch/hc676/e2_hpfft/n1024/UF1/pnr/route_status.rpt
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
puts "E2_PHASES setup=[expr $t1-$t0] synth=[expr $t2-$t1] opt=[expr $t3-$t2] place=[expr $t4-$t3] physopt=[expr $t5-$t4] route=[expr $t6-$t5]"
puts "E2 ROUTED WNS $wns"
puts "E2 IMPLEMENTATION OK"
