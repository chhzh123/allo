set fh [open files.txt]
foreach f [split [string trim [read $fh]] "\n"] { read_verilog -sv /scratch/hc676/gemmini_full/out/shift_4/$f }
close $fh
synth_design -top Gemmini -part xcu280-fsvh2892-2L-e -mode out_of_context -retiming -verilog_define SYNTHESIS
create_clock -period 3.333 -name clk [get_ports clock]
opt_design
place_design
phys_opt_design -retime
route_design
phys_opt_design -retime
report_utilization -file util.rpt
report_utilization -hierarchical -hierarchical_depth 6 -file util_hier.rpt
report_timing_summary -file timing.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
write_checkpoint -force routed.dcp
