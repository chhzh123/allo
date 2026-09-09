# E2 rerun of HP-FFT (UCLA-VAST/HP-FFT-HLS @ c4611b8) on the U280 at 300 MHz.
# Identical to the shipped common.tcl except: solution name, part/period hard-coded
# (xcu280-fsvh2892-2L-e, 3.333 ns), and export_design removed -- Vivado synthesis and
# place-and-route are run out-of-context by /scratch/hc676/e2_hpfft/run_pnr.sh instead.
set project_name "build"
set src_path "."
open_project "$src_path/$project_name"
set source_files [list "$src_path/FFT.cpp" "$src_path/testbench.cpp"]
foreach file $source_files {
    if {[file exists $file]} { add_files $file } else { puts "Warning: File $file not found!" }
}
open_solution "FFT_300MHz"
set_part xcu280-fsvh2892-2L-e
set_top "FFT_TOP"
create_clock -period 3.333
config_compile -unsafe_math_optimizations
add_files -tb "$src_path/testbench.cpp"
csim_design
csynth_design
close_project
exit
