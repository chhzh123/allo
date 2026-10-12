#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_paper.sh [I J K]: AutoSA's own published int8 matrix multiplication
# (its docs/examples/mm_int8_large.rst, the design of the FPGA'21 paper's
# table): 24 x 8 cells of 64 multipliers each, array_part[264,256,64],
# latency[11,32], simd[64], --host-serialize, on its own input program
# (autosa_tests/large/mm_int8, 1056 x 1024 x 1024 unless three sizes are
# given). Generated with the documented command, then synthesised and
# cosimulated here, to compare the cycles it takes in RTL with the kernel
# time its authors measured on a U250: 0.759 ms unoptimised, 0.303 ms at 300
# MHz after AutoBridge. Nothing is routed.
main() {
set -u
export PATH=/scratch/hc676/allo-agent/bin:/scratch/hc676/autosa/llvm9/bin:$PATH
export LD_LIBRARY_PATH=/scratch/hc676/autosa/llvm9/lib:${LD_LIBRARY_PATH:-}
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
A=/scratch/hc676/autosa/AutoSA
T=$A/autosa_tests/large/mm_int8
E=/scratch/hc676/e16_tiles
I=${1:-1056}; J=${2:-1024}; K=${3:-1024}
D=$E/autosa_paper/mm_int8_${I}x${J}x${K}
LOG=$E/logs/autosa_paper_mm_int8_${I}x${J}x${K}.log
mkdir -p $E/logs; rm -rf $D; mkdir -p $D/out/src $D/out/latency_est $D/out/resource_est $D/out/tuning
echo "AUTOSA_PAPER_START mm_int8 $I x $J x $K $(date)" > $LOG
# ---- 1. generate, by the documented command ---------------------------------
sed -E "s/^#define I [0-9]+/#define I $I/; s/^#define J [0-9]+/#define J $J/; s/^#define K [0-9]+/#define K $K/" $T/kernel.h > $D/kernel.h
cp $T/kernel.c $T/simd_info.json $D/
grep -hE "^#define [IJK] " $D/kernel.h | tr "\n" " " >> $LOG; echo >> $LOG
cd $A
./autosa $D/kernel.c --config=./autosa_config/autosa_config.json \
  --target=autosa_hls_c --output-dir=$D/out \
  --sa-sizes="{kernel[]->space_time[3];kernel[]->array_part[264,256,64];kernel[]->latency[11,32];kernel[]->simd[64]}" \
  --simd-info=$D/simd_info.json --host-serialize \
  --data-pack-sizes="{kernel[]->A[32,32,64];kernel[]->B[32,32,64];kernel[]->C[32,32,64]}" \
  --no-isl-sink --hls > $D/gen.log 2>&1
echo "gen rc=$?" >> $LOG
G=$D/out/src
[ -f $G/kernel_kernel.cpp ] || { echo "AUTOSA_PAPER_FAIL no generated design" >> $LOG; exit 1; }
echo "gen ports: $(grep -oE 'void kernel0\([^)]*' $G/kernel_kernel.cpp | head -1 | sed 's/void kernel0(//') | PE instances $(grep -cE '^ *PE_wrapper' $G/kernel_kernel.cpp)" >> $LOG
# ---- 2. csynth + cosim -------------------------------------------------------
H=$D/hls; mkdir -p $H; cd $H || exit 1
cp $G/kernel_kernel.cpp $G/kernel_kernel.h $G/kernel_host.cpp $D/kernel.h .
SIG=$(grep -oE "void kernel0\([^)]*" kernel_kernel.cpp | head -1)
beats() {
  local e
  e=$(grep -oE "\*dev_$1 = \([a-z ]*\*\)malloc\(.*\* sizeof" kernel_host.cpp | head -1 \
    | sed -E "s/.*malloc\((.*) \* sizeof/\1/" | tr -d " ()")
  echo $(( e ))
}
PA=$(echo "$SIG" | grep -oE "A_t[0-9]+" | tr -dc 0-9)
PB=$(echo "$SIG" | grep -oE "B_t[0-9]+" | tr -dc 0-9)
PC=$(echo "$SIG" | grep -oE "C_t[0-9]+" | tr -dc 0-9)
EA=$(beats A); EB=$(beats B); EC=$(beats C)
DA=$(( EA/PA )); DB=$(( EB/PB )); DC=$(( EC/PC ))
echo "hls A_t$PA B_t$PB C_t$PC, host elements A=$EA B=$EB C=$EC -> depth A=$DA B=$DB C=$DC" >> $LOG
sed -i -e "s/\(#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem_A\)[[:space:]]*$/\1 depth=$DA/" \
       -e "s/\(#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem_B\)[[:space:]]*$/\1 depth=$DB/" \
       -e "s/\(#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem_C\)[[:space:]]*$/\1 depth=$DC/" kernel_kernel.cpp
cat > cosim.tcl <<TCL
open_project cosim_prj
set_top kernel0
add_files kernel_kernel.cpp
add_files -tb kernel_host.cpp
open_solution sol
set_part xcu280-fsvh2892-2L-e
create_clock -period 3.333 -name default
config_compile -name_max_length 50
csynth_design
cosim_design -rtl verilog
exit
TCL
ulimit -s unlimited 2>/dev/null || ulimit -s 1048576
T0=$(date +%s)
timeout ${HLS_LIMIT:-259200} vitis_hls -f cosim.tcl > cosim.log 2>&1
echo "hls rc=$? wall_s=$(( $(date +%s) - T0 ))" >> $LOG
grep -E "Finished Command csynth_design|Finished Command cosim_design" cosim.log | sed "s/CPU user time.*Elapsed time: /elapsed /" | cut -c1-160 >> $LOG
grep -E "^Passed!|^Failed with|C/RTL co-simulation finished|DEADLOCK DETECTED" cosim.log | head -4 >> $LOG
grep -E "\| *kernel0 *\||Verilog" cosim_prj/sol/sim/report/*_cosim.rpt 2>/dev/null | head -3 | cut -c1-140 >> $LOG
echo "AUTOSA_PAPER_DONE $(date)" >> $LOG
}
main "$@"
