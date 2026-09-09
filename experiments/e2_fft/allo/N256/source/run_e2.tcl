# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#=============================================================================
# run.tcl 
#=============================================================================
# Project name
set hls_prj out.prj

# Open/reset the project
open_project ${hls_prj} -reset

open_solution -reset solution1 -flow_target vivado

# Top function of the design is "fft"
set_top fft

# Add design and testbench files
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=gnu++0x -DE2_N=256 -DE2_NT=4"
open_solution "solution1"

# Target device is u280
set_part {xcu280-fsvh2892-2L-e}

# Target frequency
create_clock -period 3.33

# Run HLS
csim_design
csynth_design
cosim_design -trace_level port -rtl verilog

exit
