open_project prj
set_top lanes8_r0_0
add_files kernel.cpp
open_solution sol
set_part xcu280-fsvh2892-2L-e
create_clock -period 4.75 -name default
set_clock_uncertainty 0.900
config_interface -clock_enable=0
config_compile -pipeline_loops 64
# Every multiply in a DSP. Left to itself HLS builds a lone int8 x int8 (a
# cell whose partial sum is a folded constant has no add to pair it with) as
# a 16-bit LUT multiplier, seven logic levels that were the worst path of a
# 4x4 array at a 2 ns target; the DSP does it in its own pipeline.
config_op mul -impl dsp

set_directive_interface -mode ap_ctrl_none "lanes8_r0_0" return
csynth_design
export_design -format ip_catalog
exit
