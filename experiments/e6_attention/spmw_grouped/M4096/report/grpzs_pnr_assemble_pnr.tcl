create_project -in_memory -part xcu280-fsvh2892-2L-e
set root /scratch/hc676/e6_work/grpzs_4096_pnr
add_files [glob $root/*.sv]
foreach r {mac_r0 mac_r1 mac_r2 mac_r3 mac_r4 mac_r5 act_r0} {
  set d $root/$r
  add_files [glob -nocomplain $d/*.sv]
  add_files [glob -nocomplain $d/prj/sol/syn/verilog/*.v]
  foreach x [glob -nocomplain $d/prj/sol/impl/ip/tmp.srcs/sources_1/ip/*/*.xci] {
    read_ip $x
  }
}
generate_target synthesis [get_ips]
set_property top spmw_harness [current_fileset]
add_files -fileset constrs_1 /scratch/hc676/e6_work/grpzs_4096_pnr/clock.xdc
proc stage {name body} {
  set t0 [clock milliseconds]
  uplevel 1 $body
  puts "SPMW STAGE $name [expr {([clock milliseconds] - $t0) / 1000.0}]"
}
stage synth  { synth_design -top spmw_harness -part xcu280-fsvh2892-2L-e }
report_utilization -file util_synth.rpt

stage opt    { opt_design }
stage place  { place_design  }
stage physopt { phys_opt_design  }
stage route  { route_design  }
report_utilization -file util.rpt
report_utilization -hierarchical -hierarchical_depth 2 -file util_hier.rpt
report_timing_summary -file timing.rpt
report_route_status -file route.rpt
write_checkpoint -force routed.dcp
set wns [get_property SLACK [get_timing_paths -delay_type max]]
puts "ARRAY WNS $wns"
set unrouted [llength [get_nets -quiet -filter {ROUTE_STATUS != ROUTED && ROUTE_STATUS != INTRASITE} -of [get_nets -quiet -hierarchical]]]
puts "SPMW UNROUTED $unrouted"
puts "IMPLEMENTATION OK"

