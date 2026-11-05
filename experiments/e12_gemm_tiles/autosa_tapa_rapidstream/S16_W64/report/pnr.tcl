foreach f [glob -nocomplain /scratch/hc676/e22_autosa_tapa/e12rs/rs/W64/dse/solution_0/hdl/*.v] { read_verilog $f }
foreach f [glob -nocomplain /scratch/hc676/e22_autosa_tapa/e12rs/rs/W64/dse/solution_0/hdl/*.sv] { read_verilog -sv $f }
synth_design -top kernel0 -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 3.333 -name ap_clk [get_ports ap_clk]
read_xdc /scratch/hc676/e22_autosa_tapa/e12rs/pnr/W64/floorplan.xdc
foreach pb [get_pblocks] {
  puts "PBLOCK $pb IS_SOFT [get_property IS_SOFT $pb] top_cells [llength [get_cells -quiet -of_objects $pb]] leaf_cells [llength [get_cells -quiet -hierarchical -filter "PBLOCK == $pb && IS_PRIMITIVE"]]"
}
opt_design
place_design
phys_opt_design
route_design
report_timing_summary -file timing.rpt
report_route_status -file route_status.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
write_checkpoint -force routed.dcp
report_utilization -file util.rpt
report_utilization -hierarchical -hierarchical_depth 1 -file util_hier.rpt
foreach pb [get_pblocks] { report_utilization -pblocks $pb -append -file util_pblocks.rpt }
