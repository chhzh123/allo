open_project csim_prj -reset
open_solution -reset sol -flow_target vivado
set_top top
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=gnu++0x"
foreach f [glob *.bin] { add_files -tb $f }
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.333
csim_design -O
exit
