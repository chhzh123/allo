foreach f [glob -nocomplain /scratch/hc676/e16_tiles/autosa/wide_k64_l4/S16_W256/hls/cosim_prj/sol/syn/verilog/*.v] { read_verilog $f }
foreach f [glob -nocomplain /scratch/hc676/e16_tiles/autosa/wide_k64_l4/S16_W256/hls/cosim_prj/sol/syn/verilog/*.sv] { read_verilog -sv $f }
synth_design -top kernel0 -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 3.333 -name ap_clk [get_ports ap_clk]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file util.rpt
report_utilization -hierarchical -hierarchical_depth 1 -file util_hier.rpt
report_timing_summary -file timing.rpt
report_route_status -file route_status.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
