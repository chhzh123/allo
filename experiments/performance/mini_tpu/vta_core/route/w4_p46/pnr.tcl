foreach f [glob -nocomplain /scratch/hc676/vta/hardware/chisel/vta_out_Core_w4/*.v]  { read_verilog $f }
synth_design -top Core -part xcu280-fsvh2892-2L-e -mode out_of_context -retiming
create_clock -period 4.6 -name clk [get_ports clock]
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
