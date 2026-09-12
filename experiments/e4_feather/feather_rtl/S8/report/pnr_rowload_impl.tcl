create_project -in_memory -part xcu280-fsvh2892-2L-e
add_files [glob /scratch/hc676/feather_loader/RTL_new/*.v]
add_files -fileset constrs_1 /scratch/hc676/feather_loader/pnr/rowload_N8/clock.xdc
proc stage {name body} {
  set t0 [clock milliseconds]
  uplevel 1 $body
  puts "E4 STAGE $name [expr {([clock milliseconds] - $t0) / 1000.0}]"
}
stage synth { synth_design -top feather_top -part xcu280-fsvh2892-2L-e -mode out_of_context -generic DPE_COL_NUM=8 -generic DPE_ROW_NUM=8 }
report_utilization -file /scratch/hc676/feather_loader/pnr/rowload_N8/util_synth.rpt
stage opt { opt_design }
stage place { place_design }
stage physopt { phys_opt_design }
stage route { route_design }
report_utilization -file /scratch/hc676/feather_loader/pnr/rowload_N8/util.rpt
report_utilization -hierarchical -hierarchical_depth 1 -file /scratch/hc676/feather_loader/pnr/rowload_N8/util_hier.rpt
report_timing_summary -file /scratch/hc676/feather_loader/pnr/rowload_N8/timing.rpt
report_route_status -file /scratch/hc676/feather_loader/pnr/rowload_N8/route.rpt
set wns [get_property SLACK [get_timing_paths -delay_type max]]
puts "E4 WNS $wns"
set unrouted [llength [get_nets -quiet -filter {ROUTE_STATUS != ROUTED && ROUTE_STATUS != INTRASITE} -of [get_nets -quiet -hierarchical]]]
puts "E4 UNROUTED $unrouted"
puts "IMPLEMENTATION OK"
