# The routed Gemmini accelerator, timed in the scope that has a counterpart in
# SPMW's programmable engine and in VTA's datapath and scratchpads: the execute
# controller with the mesh, and the scratchpad's memories, accumulator and
# scale units. The DMA, the TLB, the load and store controllers, the
# reservation station, the command queues and the loop unrollers are outside.
open_checkpoint routed.dcp
set scope [get_cells -hier -filter {IS_PRIMITIVE && (NAME =~ ex_controller/* || NAME =~ spad/*) && NAME !~ spad/reader* && NAME !~ spad/writer* && NAME !~ spad/buffer* && NAME !~ spad/xbar* && NAME !~ spad/widget*}]
puts "SCOPE_CELLS [llength $scope]"
report_timing -from $scope -to $scope -max_paths 3 -nworst 1 -sort_by slack -file scope_timing.rpt
set p [get_timing_paths -from $scope -to $scope -max_paths 1]
puts "SCOPE_SLACK [get_property SLACK $p]"
puts "SCOPE_FROM [get_property NAME [get_property STARTPOINT_PIN $p]]"
puts "SCOPE_TO [get_property NAME [get_property ENDPOINT_PIN $p]]"
puts "SCOPE_LEVELS [get_property LOGIC_LEVELS $p]"
set e [get_cells -hier -filter {IS_PRIMITIVE && NAME =~ ex_controller/*}]
set q [get_timing_paths -from $e -to $e -max_paths 1]
puts "EXECUTE_SLACK [get_property SLACK $q]"
puts "EXECUTE_FROM [get_property NAME [get_property STARTPOINT_PIN $q]]"
puts "EXECUTE_TO [get_property NAME [get_property ENDPOINT_PIN $q]]"
set m [get_cells -hier -filter {IS_PRIMITIVE && NAME =~ ex_controller/mesh/*}]
set r [get_timing_paths -from $m -to $m -max_paths 1]
puts "MESH_SLACK [get_property SLACK $r]"
set w [get_timing_paths -max_paths 1]
puts "WHOLE_SLACK [get_property SLACK $w]"
puts "WHOLE_FROM [get_property NAME [get_property STARTPOINT_PIN $w]]"
puts "WHOLE_TO [get_property NAME [get_property ENDPOINT_PIN $w]]"
puts "WHOLE_LEVELS [get_property LOGIC_LEVELS $w]"
