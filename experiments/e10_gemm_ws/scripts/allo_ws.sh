#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# allo_ws.sh <S>: Allo's weight-stationary S x S array on one S x S x S
# matmul, by the recipe E1 settled on for Allo (allo_beats.sh, vcd_allo.sh,
# allo_ooc.sh).
#   1. ws_allo_gen.py: check the design on LLVM, schedule it, emit the kernel
#   2. vitis_hls: C simulation with three seeds
#   3. vitis_hls: csynth and RTL cosimulation of one launch
#   4. xsim again with a waveform: first memory beat to last (beats.py)
#   5. vivado: out-of-context place and route at 3.333 ns
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1
E=/scratch/hc676/e15_ws
D=$E/allo/S$S
LOG=$E/logs/allo_S$S.log
mkdir -p $E/logs; rm -rf $D; mkdir -p $D
echo "ALLO_WS_START S=$S $(date)" > $LOG
# ---- 1. generate -----------------------------------------------------------
PYTHONPATH=/scratch/hc676/allo python3 $E/ws_allo_gen.py $S $D/gen > $D/gen.log 2>&1
echo "gen rc=$?" >> $LOG
grep -E "^WS GEN (llvm|OK|ERROR)" $D/gen.log | cut -c1-160 >> $LOG
[ -f $D/gen/ws_gen_info.json ] || { echo "ALLO_WS_FAIL no generated design" >> $LOG; exit 1; }
# ---- 2. C simulation, three seeds ------------------------------------------
H=$D/hls; mkdir -p $H; cd $H || exit 1
cp $D/gen/kernel.cpp $D/gen/kernel.h $D/gen/tb.cpp $D/gen/tb_1launch.cpp $D/gen/*.bin .
cat > csim.tcl <<TCL
open_project csim_prj -reset
open_solution -reset sol -flow_target vivado
set_top gemm
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=gnu++0x"
foreach f [glob *.bin] { add_files -tb \$f }
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.333
csim_design -O
exit
TCL
vitis_hls -f csim.tcl > csim.log 2>&1
echo "csim rc=$?" >> $LOG
grep -E "WS TB (ALL PASS|FAILED)" csim.log | head -2 >> $LOG
# ---- 3. csynth + cosim, one launch -----------------------------------------
cat > run.tcl <<TCL
open_project prj -reset
open_solution -reset sol -flow_target vivado
set_top gemm
add_files kernel.cpp
add_files -tb tb_1launch.cpp -cflags "-std=gnu++0x"
add_files -tb A0.bin
add_files -tb B0.bin
add_files -tb Cref0.bin
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.333
csynth_design
# E1's flags: without them Vitis builds a deadlock monitor while "Generating
# RTL test bench", and that step can hang on Allo's systolic designs.
cosim_design -rtl verilog -trace_level none -disable_deadlock_detection
exit
TCL
T0=$(date +%s)
timeout 14400 vitis_hls -f run.tcl > build.log 2>&1
echo "hls rc=$? wall_s=$(( $(date +%s) - T0 ))" >> $LOG
grep -E "WS TB (seed|ALL|FAILED)|co-simulation finished" build.log | head -4 | cut -c1-160 >> $LOG
grep -E "^\| *Verilog\|" prj/sol/sim/report/gemm_cosim.rpt 2>/dev/null | cut -c1-130 >> $LOG
# ---- 4. first memory beat to last ------------------------------------------
SIM=$H/prj/sol/sim/verilog
if [ -f $SIM/run_xsim.sh ]; then
  cd $SIM
  XELAB=$(grep -m1 "bin/xelab" run_xsim.sh | sed "s/-s gemm *$/-s gemm_fo -debug typical/")
  bash -c "$XELAB" > fo_xelab.log 2>&1 || echo "beats xelab failed" >> $LOG
  T=/apatb_gemm_top/AESL_inst_gemm
  { echo "open_vcd gemm_fo.vcd"
    echo "log_vcd $T/ap_clk"
    for g in 0 1 2; do for s in RVALID RREADY WVALID WREADY; do echo "log_vcd $T/m_axi_gmem${g}_$s"; done; done
    echo "run all"; echo "close_vcd"; echo "quit"; } > fo_vcd.tcl
  xsim --noieeewarnings gemm_fo -tclbatch fo_vcd.tcl > fo_xsim.log 2>&1
  python3 $E/beats.py gemm_fo.vcd \
    A:m_axi_gmem0_RVALID:m_axi_gmem0_RREADY \
    B:m_axi_gmem1_RVALID:m_axi_gmem1_RREADY \
    C:m_axi_gmem2_WVALID:m_axi_gmem2_WREADY > $D/beats.txt 2>&1
  cat $D/beats.txt >> $LOG
  rm -f gemm_fo.vcd
else
  echo "beats: no cosim snapshot" >> $LOG
fi
# ---- 5. place and route ----------------------------------------------------
VV=$H/prj/sol/syn/verilog
OUT=$D/pnr
if [ -f "$VV/gemm.v" ]; then
  rm -rf $OUT; mkdir -p $OUT; cd $OUT
  cat > pnr.tcl <<TCL
foreach f [glob -nocomplain $VV/*.v]  { read_verilog \$f }
foreach f [glob -nocomplain $VV/*.sv] { read_verilog -sv \$f }
synth_design -top gemm -part xcu280-fsvh2892-2L-e -mode out_of_context
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
echo "ALLO_WS_DONE $(date)" >> $LOG
