read_verilog /scratch/hc676/gemmini_build/out/OS_16/Mesh.v
synth_design -top Mesh -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 3.333 -name clk [get_ports clock]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file util.rpt
report_utilization -hierarchical -file util_hier.rpt
report_timing_summary -file timing.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
