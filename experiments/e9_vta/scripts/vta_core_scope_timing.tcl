# The routed VTA engine, timed without its control: instruction fetch and
# queues, decoders, semaphores, the DMA command generators and the store unit.
# What is left is the datapath and the scratchpads that feed it.
open_checkpoint routed.dcp
set scope [get_cells -hier -filter {IS_PRIMITIVE && NAME !~ fetch/* && NAME !~ *inst_q/* && NAME !~ *vmeCmd/* && NAME !~ */dec/* && NAME !~ ecounters/* && NAME !~ store/* && NAME !~ */s/* && NAME !~ */s_0/* && NAME !~ */s_1/*}]
puts "SCOPE_CELLS [llength $scope]"
report_timing -from $scope -to $scope -max_paths 3 -nworst 1 -sort_by slack -file scope_timing.rpt
set p [get_timing_paths -from $scope -to $scope -max_paths 1]
puts "SCOPE_SLACK [get_property SLACK $p]"
puts "SCOPE_FROM [get_property NAME [get_property STARTPOINT_PIN $p]]"
puts "SCOPE_TO [get_property NAME [get_property ENDPOINT_PIN $p]]"
set g [get_cells -hier -filter {IS_PRIMITIVE && (NAME =~ compute/tensorGemm/* || NAME =~ compute/tensorAlu/*)}]
set q [get_timing_paths -from $g -to $g -max_paths 1]
puts "DATAPATH_SLACK [get_property SLACK $q]"
