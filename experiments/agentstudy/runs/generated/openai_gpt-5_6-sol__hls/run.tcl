open_project prj
set_top gemm_tile
add_files design.cpp
open_solution sol
set_part xcu280-fsvh2892-2L-e
create_clock -period 3.33 -name default
config_interface -clock_enable=0
config_rtl -reset_level low
config_op mul -impl dsp
csynth_design
export_design -format ip_catalog
exit
