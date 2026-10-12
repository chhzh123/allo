#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_pub.sh <test> [I J K]: one of AutoSA's own published matrix
# multiplications (autosa_tests/large/<test>, docs/examples/<test>_large.rst;
# the three AutoSA columns of table 6 in its FPGA'21 paper), generated with
# the documented command, then synthesised and cosimulated here for the cycles
# one launch takes in RTL.
#   mm        float   13 x 16 cells of  8  array_part[260,256,512] latency[20,16] simd[8]
#   mm_int16  uint16  16 x 16 cells of 32  array_part[256,256,32]  latency[16,16] simd[32]
#   mm_int8   int8    24 x  8 cells of 64  array_part[264,256,64]  latency[11,32] simd[64]
# The sizes default to the published problem. Smaller ones (multiples of the
# array partition) run the same array on fewer tiles.
# Three things differ from the published inputs, none of them inside the scop,
# so none of them in the hardware: kernel.h is written out (the float test
# ships as int at a reduced size); the int8 operands are random instead of all
# ones, whose int8 product is 0 everywhere; the float check is relative, since
# the array adds in another order than the reference does.
# Nothing is routed here.
main() {
set -u
export PATH=/scratch/hc676/allo-agent/bin:/scratch/hc676/autosa/llvm9/bin:$PATH
export LD_LIBRARY_PATH=/scratch/hc676/autosa/llvm9/lib:${LD_LIBRARY_PATH:-}
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
A=/scratch/hc676/autosa/AutoSA
E=/scratch/hc676/e16_tiles
TEST=$1
PACK='--data-pack-sizes={kernel[]->A[32,32,64];kernel[]->B[32,32,64];kernel[]->C[32,32,64]}'
case "$TEST" in
  mm)       TYPE=float;            AP="260 256 512"; LAT="20 16"; SIMD=8;  DEF="1040 1024 1024"; EXTRA=();;
  mm_int16) TYPE="unsigned short"; AP="256 256 32";  LAT="16 16"; SIMD=32; DEF="1024 1024 1024"; EXTRA=("$PACK");;
  mm_int8)  TYPE=char;             AP="264 256 64";  LAT="11 32"; SIMD=64; DEF="1056 1024 1024"; EXTRA=("$PACK" --no-isl-sink);;
  *) echo "unknown test $TEST" >&2; exit 2;;
esac
read -r AI AJ AK <<< "$AP"; read -r L1 L2 <<< "$LAT"; read -r DI DJ DK <<< "$DEF"
I=${2:-$DI}; J=${3:-$DJ}; K=${4:-$DK}
[ $((I % AI)) = 0 ] && [ $((J % AJ)) = 0 ] && [ $((K % AK)) = 0 ] || { echo "$I x $J x $K is not whole tiles of $AP" >&2; exit 2; }
TILES=$(( (I/AI) * (J/AJ) * (K/AK) )); PER=$(( L1 * L2 * AK / SIMD ))
T=$A/autosa_tests/large/$TEST
D=$E/autosa_paper/${TEST}_${I}x${J}x${K}
LOG=$E/logs/autosa_paper_${TEST}_${I}x${J}x${K}.log
mkdir -p $E/logs; rm -rf $D; mkdir -p $D/out/src $D/out/latency_est $D/out/resource_est $D/out/tuning
echo "AUTOSA_PUB_START $TEST $I x $J x $K $(date)" > $LOG
echo "array $((AI/L1)) x $((AJ/L2)) cells of $SIMD, $TILES tiles of $PER cycles: $((TILES*PER)) cycles at one multiply-add a multiplier a cycle" >> $LOG
# ---- 1. generate, by the documented command ---------------------------------
printf '#include "stdio.h"\n#include "stdlib.h"\n#include "math.h"\ntypedef %s data_t;\n#define I %s\n#define J %s\n#define K %s\n' "$TYPE" $I $J $K > $D/kernel.h
cp $T/kernel.c $T/simd_info.json $D/
case "$TEST" in
  mm_int8) sed -i -E 's/(A\[i\]\[k\]|B\[j\]\[k\]) = 1;/\1 = rand();/' $D/kernel.c;;
  mm)      sed -i -E 's/> 0\.001\)/> 0.0001 * (1 + fabs((float)C_golden[i][j])))/' $D/kernel.c;;
esac
echo "kernel.c lines changed from the published one: $(diff $T/kernel.c $D/kernel.c | grep -c '^>')" >> $LOG
cd $A
./autosa $D/kernel.c --config=./autosa_config/autosa_config.json \
  --target=autosa_hls_c --output-dir=$D/out \
  --sa-sizes="{kernel[]->space_time[3];kernel[]->array_part[$AI,$AJ,$AK];kernel[]->latency[$L1,$L2];kernel[]->simd[$SIMD]}" \
  --simd-info=$D/simd_info.json --host-serialize \
  ${EXTRA[@]+"${EXTRA[@]}"} --hls > $D/gen.log 2>&1
echo "gen rc=$?" >> $LOG
G=$D/out/src
[ -f $G/kernel_kernel.cpp ] || { echo "AUTOSA_PUB_FAIL no generated design" >> $LOG; exit 1; }
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
echo "depth pragmas: $(grep -c 'bundle=gmem_[ABC] depth=' kernel_kernel.cpp)" >> $LOG
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
echo "AUTOSA_PUB_DONE $(date)" >> $LOG
}
main "$@"
