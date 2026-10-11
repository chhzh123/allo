#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_ws.sh <S> <narrow|wide>: AutoSA's weight-stationary S x S array on one
# S x S x S matmul, end to end, on E1's recipe with the two changes AutoSA's own
# documentation makes for this array (docs/examples/mm.rst, "Array 6: [j,k]"):
# the space-time mapping is space_time[5], whose space loops are (j, k), and
# the compiler is told the design reduces along its cells with
# --local-reduce --reduce-op="+". Under that mapping B[k][j] never leaves its
# element, A crosses the array and C is summed down it. The documentation's
# third flag, --simd-touch-space, is for vectorising along a space loop; E1's
# recipe has no vectorisation (simd[1]) and with it the generator stops after
# its analysis and writes no design. `wide` adds --host-serialize, E1's
# 512-bit variant.
#   1. autosa: generate kernel_kernel.cpp
#   2. vitis_hls: csynth and RTL cosimulation with AutoSA's own host testbench
#   3. xsim again with a waveform: first memory beat to last (beats.py)
#   4. vivado: out-of-context place and route at 3.333 ns
set -u
export PATH=/scratch/hc676/allo-agent/bin:/scratch/hc676/autosa/llvm9/bin:$PATH
export LD_LIBRARY_PATH=/scratch/hc676/autosa/llvm9/lib:${LD_LIBRARY_PATH:-}
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1; V=${2:-narrow}
A=/scratch/hc676/autosa/AutoSA
REF=/scratch/hc676/e1_autosa_2026-09-06/gen/mm8i8z
E=/scratch/hc676/e15_ws
D=$E/autosa/$V/S$S
LOG=$E/logs/autosa_${V}_S$S.log
mkdir -p $E/logs; rm -rf $D; mkdir -p $D/out/src $D/out/latency_est $D/out/resource_est $D/out/tuning
echo "AUTOSA_WS_START S=$S $V $(date)" > $LOG
# ---- 1. generate -----------------------------------------------------------
sed -E "s/#define I [0-9]+/#define I $S/; s/#define J [0-9]+/#define J $S/; s/#define K [0-9]+/#define K $S/" \
    $REF/kernel.h > $D/kernel.h
cp $REF/kernel.c $D/kernel.c; cp $REF/simd_info.json $D/
EXTRA=""; [ "$V" = wide ] && EXTRA="--host-serialize"
cd $A
./autosa $D/kernel.c --config=./autosa_config/autosa_config.json \
  --target=autosa_hls_c --output-dir=$D/out $EXTRA \
  --sa-sizes="{kernel[]->space_time[5];kernel[]->array_part[$S,$S,$S];kernel[]->latency[1,1];kernel[]->simd[1]}" \
  --simd-info=$D/simd_info.json --hls \
  --local-reduce --reduce-op="+" --array-contraction > $D/gen.log 2>&1
echo "gen rc=$?" >> $LOG
G=$D/out/src
[ -f $G/kernel_kernel.cpp ] || { echo "AUTOSA_WS_FAIL no generated design: $(grep -oE 'Error: .*' $D/gen.log | head -1)" >> $LOG; exit 1; }
echo "gen ports: $(grep -oE 'void kernel0\([^)]*' $G/kernel_kernel.cpp | head -1 | sed 's/void kernel0(//') | PE instances $(grep -cE '^ *PE_wrapper' $G/kernel_kernel.cpp)" >> $LOG
# ---- 2. csynth + cosim -----------------------------------------------------
H=$D/hls; rm -rf $H; mkdir -p $H; cd $H || exit 1
cp $G/kernel_kernel.cpp $G/kernel_kernel.h $G/kernel_host.cpp $D/kernel.h .
SIG=$(grep -oE "void kernel0\([^)]*" kernel_kernel.cpp | head -1)
PA=$(echo "$SIG" | grep -oE "A_t[0-9]+" | tr -dc 0-9)   # elements per A beat
PB=$(echo "$SIG" | grep -oE "B_t[0-9]+" | tr -dc 0-9)
PC=$(echo "$SIG" | grep -oE "C_t[0-9]+" | tr -dc 0-9)
DA=$(( S*S/PA )); [ $DA -lt 1 ] && DA=1
DB=$(( S*S/PB )); [ $DB -lt 1 ] && DB=1
DC=$(( S*S/PC )); [ $DC -lt 1 ] && DC=1
echo "hls A_t$PA B_t$PB C_t$PC -> depth A=$DA B=$DB C=$DC" >> $LOG
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
csynth_design
cosim_design -rtl verilog
exit
TCL
T0=$(date +%s)
vitis_hls -f cosim.tcl > cosim.log 2>&1
echo "hls rc=$? wall_s=$(( $(date +%s) - T0 ))" >> $LOG
grep -E "^Passed!|^Failed with|C/RTL co-simulation finished" cosim.log | head -3 >> $LOG
grep -E "\| *kernel0 *\|" cosim_prj/sol/sim/report/*_cosim.rpt 2>/dev/null | head -3 | cut -c1-140 >> $LOG
# ---- 3. first memory beat to last ------------------------------------------
SIM=$H/cosim_prj/sol/sim/verilog
if [ -f $SIM/run_xsim.sh ]; then
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
else
  echo "beats: no cosim snapshot" >> $LOG
fi
# ---- 4. place and route ----------------------------------------------------
VV=$H/cosim_prj/sol/syn/verilog
OUT=$D/pnr
if [ -d "$VV" ]; then
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
  echo "pnr rc=$? wall_s=$(( $(date +%s) - T0 ))" >> $LOG
  grep -E "^PNR_UNROUTED" pnr.log >> $LOG
  python3 - >> $LOG <<'PY'
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
else
  echo "pnr: no RTL" >> $LOG
fi
echo "AUTOSA_WS_DONE $(date)" >> $LOG
