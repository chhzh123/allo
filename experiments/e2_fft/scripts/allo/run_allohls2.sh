#!/bin/bash
# usage: run_allohls2.sh <N> <wrap|raw> -- csim + csynth + cosim of the Allo-generated strided FFT project
# (Allo's own run.tcl with the XRT host swapped for tb.cpp and csim/cosim added; kernel.cpp untouched)
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
ulimit -n 8192
A=/scratch/hc676/e2_allo; P=$A/strided_n$1_$2
cd $P || exit 1
cp $A/tb.cpp .
sed -e "s|add_files -tb host.cpp -cflags \"-std=gnu++0x\"|add_files -tb tb.cpp -cflags \"-std=gnu++0x -DE2_N=$1 -DE2_NT=4\"|" \
    -e "s|^csynth_design|csim_design\ncsynth_design\ncosim_design -trace_level port -rtl verilog|" run.tcl > run_e2.tcl
rm -f hls.done
T0=$(date +%s)
timeout -k 120 21600 vitis_hls -f run_e2.tcl > hls.log 2>&1
RC=$?
echo "rc=$RC hls_wall_s=$(( $(date +%s) - T0 )) start=$T0 end=$(date +%s)" > hls.done
