open_project prj -reset
open_solution -reset sol -flow_target vivado
set_top gemm
add_files kernel.cpp
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.333
csynth_design
exit
