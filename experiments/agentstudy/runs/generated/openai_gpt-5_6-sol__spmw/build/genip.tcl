create_project -force ipgen /scratch/hc676/agentstudy/trials/openai_gpt-5_6-sol__spmw/build/ipgen -part xcu280-fsvh2892-2L-e
foreach r {mac_r0 mac_r1 mac_r2 mac_r3 mac_r4 mac_r5 mac_r6 mac_r7 mac_r8 drain_r0 drain_r1 drain_r2} {
  foreach x [glob -nocomplain       /scratch/hc676/agentstudy/trials/openai_gpt-5_6-sol__spmw/build/$r/prj/sol/impl/ip/tmp.srcs/sources_1/ip/*/*.xci] {
    import_ip $x
  }
}
generate_target {simulation} [get_ips]
