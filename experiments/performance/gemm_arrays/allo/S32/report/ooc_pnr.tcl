foreach f [glob -nocomplain /scratch/hc676/e1_allo_beats/S32/prj/sol/syn/verilog/*.v]  { read_verilog $f }
foreach f [glob -nocomplain /scratch/hc676/e1_allo_beats/S32/prj/sol/syn/verilog/*.sv] { read_verilog -sv $f }
synth_design -top gemm -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 3.333 -name ap_clk [get_ports ap_clk]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file util.rpt
report_utilization -hierarchical -file util_hier.rpt
report_timing_summary -file timing.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
