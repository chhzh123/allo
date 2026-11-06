foreach f {spmw_fifo.sv spmw_const.sv spmw_top.sv spmw_control_s_axi.sv spmw_kernel.sv} {
  read_verilog -sv /scratch/hc676/e16_tiles/spmw/kernel_S16/$f
}
foreach f [glob -nocomplain /scratch/hc676/e16_tiles/spmw/kernel_S16/roles/*/*_r*.sv] { read_verilog -sv $f }
foreach f [glob -nocomplain /scratch/hc676/e16_tiles/spmw/kernel_S16/roles/*/prj/sol/syn/verilog/*.v] { read_verilog $f }
foreach f [glob -nocomplain /scratch/hc676/e16_tiles/spmw/kernel_S16/feeders/*/prj/sol/syn/verilog/*.v] { read_verilog $f }
synth_design -top spmw_kernel -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 3.333 -name ap_clk [get_ports ap_clk]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file util.rpt
report_utilization -hierarchical -hierarchical_depth 2 -file util_hier.rpt
report_timing_summary -file timing.rpt
report_route_status -file route_status.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
