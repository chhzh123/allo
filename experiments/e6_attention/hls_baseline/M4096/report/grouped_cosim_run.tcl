open_project -reset prj
set_top attention_pv
add_files attn_grouped.cpp
add_files -tb tb.cpp -cflags "-DGROUPED"
open_solution -reset sol -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.333 -name default
csim_design -argv "/scratch/hc676/e6_work/hls/ops/M4096_s0.txt /scratch/hc676/e6_work/hls/ops/M4096_s1.txt /scratch/hc676/e6_work/hls/ops/M4096_s2.txt"
csynth_design
cosim_design -argv "/scratch/hc676/e6_work/hls/ops/M4096_s0.txt"
exit
