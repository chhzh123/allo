#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_ws_tail.sh <S> <narrow|wide> <vitis_hls pid>: finish autosa_ws.sh's
# flow for a build whose driver was stopped while Vitis HLS was still running,
# with steps 3 and 4 of that script unchanged but step 4 started as soon as
# csynth_design has returned, so the route runs beside the cosimulation
# instead of after it. Used for 32x32, where synthesis alone took nine hours.
set -u
export PATH=/scratch/hc676/allo-agent/bin:/scratch/hc676/autosa/llvm9/bin:$PATH
export LD_LIBRARY_PATH=/scratch/hc676/autosa/llvm9/lib:${LD_LIBRARY_PATH:-}
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1; V=$2; PID=$3
E=/scratch/hc676/e15_ws
D=$E/autosa/$V/S$S
H=$D/hls
LOG=$E/logs/autosa_${V}_S$S.log
START=$(stat -c %Y $H/cosim.tcl)
echo "tail: driver replaced by autosa_ws_tail.sh $(date)" >> $LOG
# ---- 4. place and route, once the RTL is final ------------------------------
route() {
  until grep -q "Finished Command csynth_design" $H/cosim.log; do
    kill -0 $PID 2>/dev/null || return 1
    sleep 60
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
# ---- 2, the rest of it: wait for the cosimulation ---------------------------
while kill -0 $PID 2>/dev/null; do sleep 60; done
echo "hls wall_s=$(( $(date +%s) - START ))" >> $LOG
grep -E "^Passed!|^Failed with|C/RTL co-simulation finished" $H/cosim.log | head -3 >> $LOG
grep -E "\| *kernel0 *\|" $H/cosim_prj/sol/sim/report/*_cosim.rpt 2>/dev/null | head -3 | cut -c1-140 >> $LOG
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
wait $ROUTE
if [ -f $D/pnr_lines.txt ]; then cat $D/pnr_lines.txt >> $LOG; else echo "pnr: no RTL" >> $LOG; fi
echo "AUTOSA_WS_DONE $(date)" >> $LOG
