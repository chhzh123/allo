set files [glob -nocomplain /scratch/hc676/agentstudy/steady/moonshotai_kimi-k3__hls//sim/*.v /scratch/hc676/agentstudy/steady/moonshotai_kimi-k3__hls//sim/*.sv]
foreach f $files { if {[string match *tb_study* $f]} { continue }
  if {[string match *.sv $f]} { read_verilog -sv $f } else { read_verilog $f } }
synth_design -top dut_norm -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 3.333 -name ap_clk [get_ports ap_clk]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file util.rpt
report_timing_summary -file timing.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
