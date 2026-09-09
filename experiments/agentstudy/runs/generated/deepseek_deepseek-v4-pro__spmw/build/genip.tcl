create_project -force ipgen /scratch/hc676/agentstudy/trials/deepseek_deepseek-v4-pro__spmw/build/ipgen -part xcu280-fsvh2892-2L-e
foreach r {pe_r0 pe_r1 pe_r2 pe_r3 pe_r4 pe_r5 pe_r6 pe_r7 pe_r8} {
  foreach x [glob -nocomplain       /scratch/hc676/agentstudy/trials/deepseek_deepseek-v4-pro__spmw/build/$r/prj/sol/impl/ip/tmp.srcs/sources_1/ip/*/*.xci] {
    import_ip $x
  }
}
generate_target {simulation} [get_ips]
