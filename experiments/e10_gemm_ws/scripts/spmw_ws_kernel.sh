#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# spmw_ws_kernel.sh <S> [jobs] [design] [tag]: an SPMW array as a kernel -- by
# default the weight-stationary mesh `gemmws`, with a memory mover on each of
# its three edge bindings and a control register file -- simulated against an
# AXI memory holding one launch's operands (E1's e1_memsim2.sh), then placed
# and routed out of context at 3.333 ns by the recipe the AutoSA and Allo rows
# use, movers and control included. `autosa e1kernel` does the same for E1's
# output-stationary kernel, whose own route left the movers out.
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export PYTHONPATH=/scratch/hc676/allo:/scratch/hc676/allo/tests/dataflow/spmw
S=$1; J=${2:-4}; DESIGN=${3:-gemmws}; TAG=${4:-kernel}
E=/scratch/hc676/e15_ws; W=$E/spmw; mkdir -p $W $E/logs
LOG=$E/logs/spmw_${TAG}_S$S.log
cd /scratch/hc676/allo
echo "SPMW_WS_KERNEL_START $DESIGN S=$S $(date)" > $LOG

K=$W/${TAG}_S$S; rm -rf $K ${K}_sim ${K}_ops ${K}_pnr
T0=$(date +%s)
timeout 5400 python3 -u scripts/spmw_package_kernel.py --design $DESIGN --size $S \
  --frequency 300 --jobs $J --sim --out $K > $K.package.log 2>&1
echo "package rc=$? wall_s=$(( $(date +%s) - T0 ))" >> $LOG
grep -hE "^HLS:|AXI data widths|master" $K.package.log | head -4 | cut -c1-200 >> $LOG
timeout 1200 python3 -u scripts/spmw_array_operands.py --design $DESIGN --size $S \
  --out ${K}_ops > $K.ops.log 2>&1
echo "operands rc=$?" >> $LOG
T0=$(date +%s)
timeout 7200 python3 -u scripts/spmw_kernel_sim.py $K ${K}_ops --design $DESIGN \
  --size $S > $K.sim.log 2>&1
echo "sim rc=$? wall_s=$(( $(date +%s) - T0 ))" >> $LOG
grep -hE "SPMW TB|xsim:" $K.sim.log | cut -c1-200 >> $LOG

OUT=${K}_pnr; mkdir -p $OUT; cd $OUT
cat > pnr.tcl <<TCL
foreach f {spmw_fifo.sv spmw_const.sv spmw_top.sv spmw_control_s_axi.sv spmw_kernel.sv} {
  read_verilog -sv $K/\$f
}
foreach f [glob -nocomplain $K/roles/*/*_r*.sv] { read_verilog -sv \$f }
foreach f [glob -nocomplain $K/roles/*/prj/sol/syn/verilog/*.v] { read_verilog \$f }
foreach f [glob -nocomplain $K/feeders/*/prj/sol/syn/verilog/*.v] { read_verilog \$f }
synth_design -top spmw_kernel -part xcu280-fsvh2892-2L-e -mode out_of_context
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
echo "SPMW_WS_KERNEL_DONE $(date)" >> $LOG
