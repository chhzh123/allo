#!/bin/bash
# Synthesise and cosimulate one wide-interface matched AutoSA design.
# Same recipe as hls_matched.sh; only the port depths change, because the
# operands now arrive 64 bytes a beat instead of S bytes.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1
G=/scratch/hc676/e1_autosa_wide/S$S/out/src
H=/scratch/hc676/e1_autosa_wide/hls/S$S
rm -rf $H; mkdir -p $H; cd $H || exit 1
[ -f $G/kernel_kernel.cpp ] || { echo "S=$S no generated design"; exit 1; }
cp $G/kernel_kernel.cpp $G/kernel_kernel.h $G/kernel_host.cpp /scratch/hc676/e1_autosa_wide/S$S/kernel.h .
# Beats, not elements: the serialized A/B are S*S bytes and C is S*S ints.
SIG=$(grep -oE "void kernel0\([^)]*" kernel_kernel.cpp | head -1)
PB=$(echo "$SIG" | grep -oE "A_t[0-9]+" | tr -dc 0-9)   # bytes per A beat
CB=$(echo "$SIG" | grep -oE "C_t[0-9]+" | tr -dc 0-9)   # ints  per C beat
DA=$(( S*S/PB )); [ $DA -lt 1 ] && DA=1
DC=$(( S*S/CB )); [ $DC -lt 1 ] && DC=1
echo "S=$S A_t$PB B_t$PB C_t$CB -> depth A=$DA B=$DA C=$DC"
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
echo "S=$S rc=$? wall_s=$(( $(date +%s) - T0 ))"
grep -E "^Passed!|^Failed with|C/RTL co-simulation finished" cosim.log | head -3
grep -E "\| *kernel0 *\|" cosim_prj/sol/sim/report/*_cosim.rpt 2>/dev/null | head -3 | cut -c1-140
echo "WIDE_HLS_DONE_$S"
