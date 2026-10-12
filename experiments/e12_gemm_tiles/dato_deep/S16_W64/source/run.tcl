open_project prj -reset
open_solution -reset sol -flow_target vivado
set_top top
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=gnu++0x"
foreach f [glob *.bin] { add_files -tb $f }
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.333
csynth_design
# E1's flags for Allo: without them Vitis builds a deadlock monitor while
# "Generating RTL test bench", and that step can hang on Allo's systolic designs.
cosim_design -rtl verilog -trace_level none -disable_deadlock_detection
exit
