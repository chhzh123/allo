#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_tiles.sh <S> <W> <narrow|wide>: AutoSA's output-stationary S x S array
# on a W x W x W matmul, end to end, on E1's recipe: space_time[3], one cell a
# multiplier. `wide` adds --host-serialize, E1's 512-bit variant, in which the
# host lays the operands out in the order the array reads them.
#
# KTILE (default W) is the array partition's third factor, the steps of K a
# tile holds: array_part[S, S, KTILE]. AutoSA's own examples always tile K
# (16 of 64 in docs/examples/mm.rst; 32, 64 and 512 of 1,024 in its large
# designs), and --host-serialize needs it: with K left whole the array keeps
# a row tile's A across the column tiles, the generated reader takes one copy
# a row tile and the generated serializer hands over one a tile, and the
# design fails its own C testbench. A build with KTILE set goes to
# autosa/<variant>_k<KTILE>/.
#   1. autosa: generate kernel_kernel.cpp
#   2. vitis_hls: csynth and RTL cosimulation with AutoSA's own host testbench
#   3. xsim again with a waveform: first memory beat to last (beats.py); only
#      up to BEATS_MAX (default 256) on a side, a second run of a long
#      simulation being a second day
#   4. vivado: out-of-context place and route at 3.333 ns, beside step 2's
#      cosimulation
main() {
set -u
export PATH=/scratch/hc676/allo-agent/bin:/scratch/hc676/autosa/llvm9/bin:$PATH
export LD_LIBRARY_PATH=/scratch/hc676/autosa/llvm9/lib:${LD_LIBRARY_PATH:-}
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1; W=$2; V=${3:-wide}
KT=${KTILE:-$W}; TAG=""; [ $KT = $W ] || TAG=_k$KT
A=/scratch/hc676/autosa/AutoSA
REF=/scratch/hc676/e1_autosa_2026-09-06/gen/mm8i8z
E=/scratch/hc676/e16_tiles
D=$E/autosa/$V$TAG/S${S}_W$W
LOG=$E/logs/autosa_${V}${TAG}_S${S}_W$W.log
mkdir -p $E/logs; rm -rf $D; mkdir -p $D/out/src $D/out/latency_est $D/out/resource_est $D/out/tuning
echo "AUTOSA_TILES_START S=$S W=$W $V K tile $KT $(date)" > $LOG
# ---- 1. generate -----------------------------------------------------------
sed -E "s/#define I [0-9]+/#define I $W/; s/#define J [0-9]+/#define J $W/; s/#define K [0-9]+/#define K $W/" \
    $REF/kernel.h > $D/kernel.h
# E1's program with operands that span int8 instead of 0..7.
sed -E "s/\(data_t\)\(rand\(\) % 8\)/(data_t)(rand() % 256 - 128)/" $REF/kernel.c > $D/kernel.c
grep -c "rand() % 256 - 128" $D/kernel.c | sed "s/^/operand lines rewritten: /" >> $LOG
cp $REF/simd_info.json $D/
EXTRA=""; [ "$V" = wide ] && EXTRA="--host-serialize"
cd $A
./autosa $D/kernel.c --config=./autosa_config/autosa_config.json \
  --target=autosa_hls_c --output-dir=$D/out $EXTRA \
  --sa-sizes="{kernel[]->space_time[3];kernel[]->array_part[$S,$S,$KT];kernel[]->latency[1,1];kernel[]->simd[1]}" \
  --simd-info=$D/simd_info.json --hls > $D/gen.log 2>&1
echo "gen rc=$?" >> $LOG
G=$D/out/src
[ -f $G/kernel_kernel.cpp ] || { echo "AUTOSA_TILES_FAIL no generated design: $(grep -oE 'Error: .*' $D/gen.log | head -1)" >> $LOG; exit 1; }
echo "gen ports: $(grep -oE 'void kernel0\([^)]*' $G/kernel_kernel.cpp | head -1 | sed 's/void kernel0(//') | PE instances $(grep -cE '^ *PE_wrapper' $G/kernel_kernel.cpp)" >> $LOG
# ---- 2. csynth + cosim -----------------------------------------------------
H=$D/hls; rm -rf $H; mkdir -p $H; cd $H || exit 1
cp $G/kernel_kernel.cpp $G/kernel_kernel.h $G/kernel_host.cpp $D/kernel.h .
SIG=$(grep -oE "void kernel0\([^)]*" kernel_kernel.cpp | head -1)
# A port's depth is the beats the host hands it: the elements the host
# allocates for it over the elements one beat carries.
beats() {  # <array> : elements the host allocates
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
grep -c "depth=" kernel_kernel.cpp | sed "s/^/depth pragmas: /" >> $LOG
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
# AutoSA's host program keeps its operands, its result and its golden result
# on the stack: 10 MB at 1024 on a side, over the 8 MB a process gets.
ulimit -s unlimited 2>/dev/null || ulimit -s 1048576
START=$(date +%s)
vitis_hls -f cosim.tcl > cosim.log 2>&1 &
HLS=$!
# ---- 4. place and route, once the RTL is final ------------------------------
route() {
  until grep -q "Finished Command csynth_design" $H/cosim.log 2>/dev/null; do
    kill -0 $HLS 2>/dev/null || return 1
    sleep 30
  done
  echo "csynth wall_s=$(( $(date +%s) - START ))" >> $LOG
  VV=$H/cosim_prj/sol/syn/verilog
  OUT=$D/pnr
  rm -rf $OUT; mkdir -p $OUT; cd $OUT
  cat > pnr.tcl <<TCL
foreach f [glob -nocomplain $VV/*.v] { read_verilog \$f }
foreach f [glob -nocomplain $VV/*.sv] { read_verilog -sv \$f }
synth_design -top kernel0 -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 3.333 -name ap_clk [get_ports ap_clk]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file util.rpt
report_utilization -hierarchical -hierarchical_depth 1 -file util_hier.rpt
report_timing_summary -file timing.rpt
report_route_status -file route_status.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
TCL
  T0=$(date +%s)
  vivado -mode batch -source pnr.tcl -nojournal -log pnr.log > /dev/null 2>&1
  echo "pnr rc=$? wall_s=$(( $(date +%s) - T0 ))" > $D/pnr_lines.txt
  grep -E "^PNR_UNROUTED" pnr.log >> $D/pnr_lines.txt
  python3 - >> $D/pnr_lines.txt <<'PY'
import os, re
u = open("util.rpt", errors="replace").read() if os.path.isfile("util.rpt") else ""
t = open("timing.rpt", errors="replace").read() if os.path.isfile("timing.rpt") else ""
def cell(name):
    m = re.search(r"^\|\s*%s\*?\s*\|\s*([\d.]+)\s*\|" % re.escape(name), u, re.M)
    return m.group(1) if m else "-"
m = re.search(r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)", t, re.S)
print("pnr LUT %s FF %s DSP %s BRAM %s WNS %s" %
      (cell("CLB LUTs"), cell("CLB Registers"), cell("DSPs"),
       cell("Block RAM Tile"), m.group(1) if m else "-"))
PY
}
route &
ROUTE=$!
wait $HLS
echo "hls rc=$? wall_s=$(( $(date +%s) - START ))" >> $LOG
grep -E "^Passed!|^Failed with|C/RTL co-simulation finished|DEADLOCK DETECTED" $H/cosim.log | head -4 >> $LOG
grep -E "\| *kernel0 *\|" $H/cosim_prj/sol/sim/report/*_cosim.rpt 2>/dev/null | head -3 | cut -c1-140 >> $LOG
# ---- 3. first memory beat to last ------------------------------------------
SIM=$H/cosim_prj/sol/sim/verilog
if [ ! -f $SIM/run_xsim.sh ]; then
  echo "beats: no cosim snapshot" >> $LOG
elif [ $W -gt ${BEATS_MAX:-256} ]; then
  echo "beats: not rerun at this size" >> $LOG
else
  cd $SIM
  XELAB=$(grep -m1 "bin/xelab" run_xsim.sh | sed "s/-s kernel0 *$/-s kernel0_fo -debug typical/")
  bash -c "$XELAB" > fo_xelab.log 2>&1 || echo "beats xelab failed" >> $LOG
  T=/apatb_kernel0_top/AESL_inst_kernel0
  { echo "open_vcd kernel0_fo.vcd"
    for s in ap_clk m_axi_gmem_A_RVALID m_axi_gmem_A_RREADY m_axi_gmem_B_RVALID \
             m_axi_gmem_B_RREADY m_axi_gmem_C_WVALID m_axi_gmem_C_WREADY; do
      echo "log_vcd $T/$s"
    done
    echo "run all"; echo "close_vcd"; echo "quit"; } > fo_vcd.tcl
  xsim --noieeewarnings kernel0_fo -tclbatch fo_vcd.tcl > fo_xsim.log 2>&1
  python3 $E/beats.py kernel0_fo.vcd > $D/beats.txt 2>&1
  cat $D/beats.txt >> $LOG
  rm -f kernel0_fo.vcd
fi
wait $ROUTE
if [ -f $D/pnr_lines.txt ]; then cat $D/pnr_lines.txt >> $LOG; else echo "pnr: no RTL" >> $LOG; fi
echo "AUTOSA_TILES_DONE $(date)" >> $LOG
}
main "$@"
