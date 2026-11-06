set t0 [clock seconds]
create_project -force pnr /scratch/hc676/e14_fft/hpfft/pnr/n1024_UF2_base/proj -part xcu280-fsvh2892-2L-e
set_property target_language Verilog [current_project]
add_files [glob /scratch/hc676/e14_fft/hpfft/n1024/UF2_base/build/FFT_300MHz/syn/verilog/*.v]
add_files -fileset constrs_1 /scratch/hc676/e14_fft/hpfft/pnr/n1024_UF2_base/clock.xdc
foreach ipt [glob -nocomplain /scratch/hc676/e14_fft/hpfft/n1024/UF2_base/build/FFT_300MHz/syn/verilog/*_ip.tcl] { source $ipt }
set_property top FFT_TOP [current_fileset]
update_compile_order -fileset sources_1
set t1 [clock seconds]
synth_design -top FFT_TOP -part xcu280-fsvh2892-2L-e -mode out_of_context
set t2 [clock seconds]
report_utilization -file /scratch/hc676/e14_fft/hpfft/pnr/n1024_UF2_base/util_synth.rpt
opt_design
place_design
phys_opt_design
route_design
set t3 [clock seconds]
report_utilization -file /scratch/hc676/e14_fft/hpfft/pnr/n1024_UF2_base/util.rpt
report_utilization -hierarchical -hierarchical_depth 2 -file /scratch/hc676/e14_fft/hpfft/pnr/n1024_UF2_base/util_hier.rpt
report_timing_summary -file /scratch/hc676/e14_fft/hpfft/pnr/n1024_UF2_base/timing.rpt
report_route_status -file /scratch/hc676/e14_fft/hpfft/pnr/n1024_UF2_base/route_status.rpt
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
puts "PNR_PHASES setup=[expr $t1-$t0] synth=[expr $t2-$t1] impl=[expr $t3-$t2]"
puts "PNR ROUTED WNS $wns"
puts "PNR IMPLEMENTATION OK"
