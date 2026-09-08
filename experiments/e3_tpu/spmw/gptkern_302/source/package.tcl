create_project -in_memory -part xcu280-fsvh2892-2L-e
set_property XPM_LIBRARIES {XPM_CDC XPM_MEMORY XPM_FIFO} [current_project]
add_files -norecurse [glob /scratch/hc676/gptkern_302/*.sv]
proc add_if_any {pattern} {
  # `add_files` on an empty glob is an error, not a no-op.
  set found [glob -nocomplain $pattern]
  if {[llength $found]} { add_files -norecurse $found }
}
foreach base [list /scratch/hc676/gptkern_302/feeders /scratch/hc676/gptkern_302/roles] {
  foreach d [glob -nocomplain -directory $base *] {
    # The unit's own wrapper as well as the netlist Vitis exported: the
    # wrapper is what names the ports the fabric connects to.
    add_if_any $d/*.sv
    add_if_any $d/prj/sol/syn/verilog/*.v
  }
}
set_property top spmw_kernel [current_fileset]
update_compile_order -fileset sources_1
ipx::package_project -root_dir /scratch/hc676/gptkern_302/ip -vendor allo -library spmw \
  -taxonomy /UserIP -import_files -force
set core [ipx::current_core]
set_property core_revision 1 $core
set_property supported_families {virtexuplusHBM Production} $core
# Every AXI interface has to be named as belonging to `ap_clk`, or the system
# linker gets an empty object list when it tries to connect them and stops with
# "Invalid option value '' specified for 'objects'" -- which says nothing about
# clocks at all.
foreach busif {m_axi_gmem0 m_axi_gmem1 m_axi_gmem2 m_axi_gmem3 m_axi_gmem4 m_axi_gmem5 s_axi_control} {
  ipx::associate_bus_interfaces -busif $busif -clock ap_clk $core
}
set clk [ipx::get_bus_interfaces ap_clk -of_objects $core]
set p [ipx::add_bus_parameter ASSOCIATED_RESET $clk]
set_property value ap_rst_n $p
set rst [ipx::get_bus_interfaces ap_rst_n -of_objects $core]
set p [ipx::add_bus_parameter POLARITY $rst]
set_property value ACTIVE_LOW $p

# Without these two, `package_xo` treats the IP as an ordinary one and
# re-derives its metadata, which silently drops ASSOCIATED_BUSIF -- and then
# the system linker cannot find a clock for the masters.
set_property sdx_kernel true $core
set_property sdx_kernel_type rtl $core
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core
package_xo -force -xo_path /scratch/hc676/gptkern_302/spmw_kernel.xo -kernel_name spmw_kernel \
  -ctrl_protocol ap_ctrl_hs -ip_directory /scratch/hc676/gptkern_302/ip -kernel_xml /scratch/hc676/gptkern_302/kernel.xml
exit
