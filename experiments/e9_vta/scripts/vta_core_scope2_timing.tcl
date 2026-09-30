# As scope_timing.tcl, and without the input and output buffers either: the
# datapath with only the weight, accumulator and micro-op memories -- the
# storage SPMW and Gemmini keep inside their arrays.
open_checkpoint routed.dcp
set scope [get_cells -hier -filter {IS_PRIMITIVE && NAME !~ fetch/* && NAME !~ *inst_q/* && NAME !~ *vmeCmd/* && NAME !~ */dec/* && NAME !~ ecounters/* && NAME !~ store/* && NAME !~ */s/* && NAME !~ */s_0/* && NAME !~ */s_1/* && NAME !~ load/tensorLoad_0/*}]
report_timing -from $scope -to $scope -max_paths 3 -nworst 1 -sort_by slack -file scope2_timing.rpt
set p [get_timing_paths -from $scope -to $scope -max_paths 1]
puts "SCOPE2_SLACK [get_property SLACK $p]"
puts "SCOPE2_FROM [get_property NAME [get_property STARTPOINT_PIN $p]]"
puts "SCOPE2_TO [get_property NAME [get_property ENDPOINT_PIN $p]]"
