open_project cosim_prj
set_top kernel0
add_files kernel_kernel.cpp
add_files -tb kernel_host.cpp
open_solution sol
set_part xcu280-fsvh2892-2L-e
create_clock -period 3.333 -name default
csynth_design
cosim_design -rtl verilog
exit
