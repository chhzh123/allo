#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# dato_tiles.sh <S> <W> [paper|test]: the Dato paper's daisy-chain systolic
# array (dato_gen.py), S x S, on a W x W x W matmul, by the recipe the Allo
# rows of E1 and E10 use.
#   1. dato_gen.py: check the design on Allo's dataflow simulator, emit the kernel
#   2. vitis_hls: C simulation of the launch
#   3. vitis_hls: csynth and RTL cosimulation of the launch
#   4. xsim again with a waveform: first memory beat to last (beats.py); only
#      up to BEATS_MAX (default 256) on a side
#   5. vivado: out-of-context place and route at 3.333 ns, beside step 3's
#      cosimulation
# HLS_LIMIT (seconds, default a day) bounds step 3. PIPELINE_LOOPS, if set, is
# Vitis's `config_compile -pipeline_loops`: the design carries no pipeline
# pragma, and the tool pipelines a loop itself only above 64 steps. At 64 on a
# side the cell's loop is exactly 64, so the tool pipelines the tile loop
# around it and unrolls the 64 steps in every cell; 32 makes it treat that loop
# as it treats the longer ones.
main() {
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1; W=$2; KIND=${3:-paper}
E=/scratch/hc676/e16_tiles
D=$E/dato/$KIND/S${S}_W$W
LOG=$E/logs/dato_${KIND}_S${S}_W$W.log
mkdir -p $E/logs; rm -rf $D; mkdir -p $D
echo "DATO_TILES_START S=$S W=$W $KIND $(date)" > $LOG
# ---- 1. generate -----------------------------------------------------------
PYTHONPATH=/scratch/hc676/allo python3 $E/dato_gen.py $S $W $D/gen $KIND > $D/gen.log 2>&1
echo "gen rc=$?" >> $LOG
grep -E "^DATO GEN" $D/gen.log | cut -c1-160 >> $LOG
[ -f $D/gen/kernel.cpp ] || { echo "DATO_TILES_FAIL no generated design" >> $LOG; exit 1; }
# ---- 2. C simulation -------------------------------------------------------
H=$D/hls; mkdir -p $H; cd $H || exit 1
cp $D/gen/kernel.cpp $D/gen/kernel.h $D/gen/*.bin .
sed -e "s/@S@/$S/g; s/@W@/$W/g; s/@BITS@/$(( S*8 ))/g" $E/dato_tb.cpp > tb.cpp
# A port's depth is the words the launch moves on it.
sed -i -E "s/(#pragma HLS interface m_axi port=v[0-9]+ offset=slave bundle=gmem[012])[[:space:]]*$/\1 depth=$(( W*W/S ))/" kernel.cpp
echo "depth pragmas: $(grep -c "m_axi.*depth=" kernel.cpp), kernels: $(grep -cE "^void gemm_[0-9]+_[0-9]+_" kernel.cpp)" >> $LOG
cat > csim.tcl <<TCL
open_project csim_prj -reset
open_solution -reset sol -flow_target vivado
set_top top
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=gnu++0x"
foreach f [glob *.bin] { add_files -tb \$f }
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.333
csim_design -O
exit
TCL
timeout 14400 vitis_hls -f csim.tcl > csim.log 2>&1
echo "csim rc=$?" >> $LOG
grep -E "DATO TB (PASS|FAIL|ERROR)" csim.log | head -2 >> $LOG
# ---- 3. csynth + cosim -----------------------------------------------------
CONFIG=""
if [ -n "${PIPELINE_LOOPS:-}" ]; then
  CONFIG="config_compile -pipeline_loops $PIPELINE_LOOPS"
  echo "$CONFIG" >> $LOG
fi
cat > run.tcl <<TCL
open_project prj -reset
open_solution -reset sol -flow_target vivado
set_top top
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=gnu++0x"
foreach f [glob *.bin] { add_files -tb \$f }
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.333
$CONFIG
csynth_design
# E1's flags for Allo: without them Vitis builds a deadlock monitor while
# "Generating RTL test bench", and that step can hang on Allo's systolic designs.
cosim_design -rtl verilog -trace_level none -disable_deadlock_detection
exit
TCL
START=$(date +%s)
timeout ${HLS_LIMIT:-86400} vitis_hls -f run.tcl > build.log 2>&1 &
HLS=$!
# ---- 5. place and route, once the RTL is final ------------------------------
route() {
  until grep -q "Finished Command csynth_design" $H/build.log 2>/dev/null; do
    kill -0 $HLS 2>/dev/null || return 1
    sleep 30
  done
  echo "csynth wall_s=$(( $(date +%s) - START ))" >> $LOG
  VV=$H/prj/sol/syn/verilog
  OUT=$D/pnr
  rm -rf $OUT; mkdir -p $OUT; cd $OUT
  cat > pnr.tcl <<TCL
foreach f [glob -nocomplain $VV/*.v]  { read_verilog \$f }
foreach f [glob -nocomplain $VV/*.sv] { read_verilog -sv \$f }
synth_design -top top -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 3.333 -name ap_clk [get_ports ap_clk]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file util.rpt
report_utilization -hierarchical -hierarchical_depth 2 -file util_hier.rpt
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
grep -E "DATO TB (PASS|FAIL|ERROR)|co-simulation finished" $H/build.log | head -4 | cut -c1-160 >> $LOG
grep -E "^\| *Verilog\|" $H/prj/sol/sim/report/top_cosim.rpt 2>/dev/null | cut -c1-130 >> $LOG
# ---- 4. first memory beat to last ------------------------------------------
SIM=$H/prj/sol/sim/verilog
if [ ! -f $SIM/run_xsim.sh ]; then
  echo "beats: no cosim snapshot" >> $LOG
elif [ $W -gt ${BEATS_MAX:-256} ]; then
  echo "beats: not rerun at this size" >> $LOG
else
  cd $SIM
  XELAB=$(grep -m1 "bin/xelab" run_xsim.sh | sed "s/-s top *$/-s top_fo -debug typical/")
  bash -c "$XELAB" > fo_xelab.log 2>&1 || echo "beats xelab failed" >> $LOG
  T=/apatb_top_top/AESL_inst_top
  { echo "open_vcd top_fo.vcd"
    echo "log_vcd $T/ap_clk"
    for g in 0 1 2; do for s in RVALID RREADY WVALID WREADY; do echo "log_vcd $T/m_axi_gmem${g}_$s"; done; done
    echo "run all"; echo "close_vcd"; echo "quit"; } > fo_vcd.tcl
  xsim --noieeewarnings top_fo -tclbatch fo_vcd.tcl > fo_xsim.log 2>&1
  python3 $E/beats.py top_fo.vcd \
    A:m_axi_gmem0_RVALID:m_axi_gmem0_RREADY \
    B:m_axi_gmem1_RVALID:m_axi_gmem1_RREADY \
    C:m_axi_gmem2_WVALID:m_axi_gmem2_WREADY > $D/beats.txt 2>&1
  cat $D/beats.txt >> $LOG
  rm -f top_fo.vcd
fi
wait $ROUTE
if [ -f $D/pnr_lines.txt ]; then cat $D/pnr_lines.txt >> $LOG; else echo "pnr: no RTL" >> $LOG; fi
echo "DATO_TILES_DONE $(date)" >> $LOG
}
main "$@"
