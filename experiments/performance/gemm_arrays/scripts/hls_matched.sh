#!/bin/bash
# Synthesise and cosimulate one matched AutoSA design: an S x S array on S*S*S.
# Same recipe as the fixed-problem runs; only the port depths change with S,
# because A and B arrive S elements a beat and C leaves one at a time.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1
G=/scratch/hc676/e1_autosa_matched/S$S/out/src
H=/scratch/hc676/e1_autosa_matched/hls/S$S
rm -rf $H; mkdir -p $H; cd $H || exit 1
[ -f $G/kernel_kernel.cpp ] || { echo "S=$S no generated design"; exit 1; }
cp $G/kernel_kernel.cpp $G/kernel_kernel.h $G/kernel_host.cpp /scratch/hc676/e1_autosa_matched/S$S/kernel.h .
DA=$S; DC=$((S*S))
sed -i -e "s/\(#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem_A\)[[:space:]]*$/\1 depth=$DA/" \
       -e "s/\(#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem_B\)[[:space:]]*$/\1 depth=$DA/" \
       -e "s/\(#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem_C\)[[:space:]]*$/\1 depth=$DC/" kernel_kernel.cpp
cat > cosim.tcl <<TCL
open_project cosim_prj
set_top kernel0
add_files kernel_kernel.cpp
add_files -tb kernel_host.cpp
open_solution sol
set_part xcu280-fsvh2892-2L-e
create_clock -period 3.333 -name default
csynth_design
cosim_design -rtl verilog
exit
TCL
T0=$(date +%s)
vitis_hls -f cosim.tcl > cosim.log 2>&1
RC=$?
echo "S=$S rc=$RC wall_s=$(( $(date +%s) - T0 ))"
grep -E "^Passed!|^Failed with|C/RTL co-simulation finished" cosim.log | head -3
awk '/Cosim Result/,/^$/' cosim.log 2>/dev/null | head -8
grep -E "\| *kernel0 *\|" cosim_prj/sol/sim/report/*_cosim.rpt 2>/dev/null | head -3 | cut -c1-140
echo "MATCHED_HLS_DONE_$S"
