create_project -in_memory -part xcu280-fsvh2892-2L-e
set root /scratch/hc676/e13_mem/b_ptpumem-micro_16
add_files [glob $root/*.sv]
foreach r {pe_r0 pe_r1 pe_r2 pe_r3 pe_r4 pe_r5 pe_r6 pe_r7 pe_r8 etap_r0 etap_r1 etap_r2 req16_r0 deal16_r0 head16_r0 uq_r0 tap_r0 tap_r1 tap_r2 lane_r0 ctap_r0 ctap_r1 ctap_r2 wreq16_r0 pack16_r0} {
  set d $root/$r
  add_files [glob -nocomplain $d/*.sv]
  add_files [glob -nocomplain $d/prj/sol/syn/verilog/*.v]
  foreach x [glob -nocomplain $d/prj/sol/impl/ip/tmp.srcs/sources_1/ip/*/*.xci] {
    read_ip $x
  }
}
generate_target synthesis [get_ips]
set_property top spmw_harness [current_fileset]
add_files -fileset constrs_1 /scratch/hc676/e13_mem/route_16_p30/clock.xdc
proc stage {name body} {
  set t0 [clock milliseconds]
  uplevel 1 $body
  puts "SPMW STAGE $name [expr {([clock milliseconds] - $t0) / 1000.0}]"
}
stage synth  { synth_design -top spmw_harness -part xcu280-fsvh2892-2L-e -max_dsp 0 }
report_utilization -file util_synth.rpt

stage opt    { opt_design }
stage place  { place_design  }
stage physopt { phys_opt_design  }
stage route  { route_design  }
report_utilization -file util.rpt
report_timing_summary -file timing.rpt
report_route_status -file route.rpt
write_checkpoint -force routed.dcp
set wns [get_property SLACK [get_timing_paths -delay_type max]]
puts "ARRAY WNS $wns"
set unrouted [llength [get_nets -quiet -filter {ROUTE_STATUS != ROUTED && ROUTE_STATUS != INTRASITE} -of [get_nets -quiet -hierarchical]]]
puts "SPMW UNROUTED $unrouted"
puts "IMPLEMENTATION OK"

