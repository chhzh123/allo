# Vitis HLS synthesis script for Allo-generated FFT-256 kernel
# Run with: vitis_hls -f run_hls.tcl

open_project fft_256_prj
set_top fft_256
add_files kernel.cpp
open_solution "solution1" -flow_target vivado
set_part {xcu250-figd2104-2L-e}
create_clock -period 4.0 -name default
csynth_design
close_project
