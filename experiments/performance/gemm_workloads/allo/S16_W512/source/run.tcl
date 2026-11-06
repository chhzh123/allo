open_project prj -reset
open_solution -reset sol -flow_target vivado
set_top gemm
add_files kernel.cpp
add_files -tb tb_1launch.cpp -cflags "-std=gnu++0x"
add_files -tb A0.bin
add_files -tb B0.bin
add_files -tb Cref0.bin
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.333
csynth_design
# E1's flags: without them Vitis builds a deadlock monitor while "Generating
# RTL test bench", and that step can hang on Allo's systolic designs.
cosim_design -rtl verilog -trace_level none -disable_deadlock_detection
exit
