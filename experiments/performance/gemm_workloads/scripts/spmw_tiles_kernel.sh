#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# spmw_tiles_kernel.sh <S> package [jobs] | sim <W> | pnr: SPMW's S x S
# output-stationary array as a kernel -- a memory mover on each of its three
# edge bindings and a control register file.
#   package  stages and synthesises the kernel, once a size. It is elaborated
#            at W = 256, the smallest of the workloads at which every edge
#            buffer reaches the depth a longer launch gives it; the kernel
#            reads its counts from its arguments, so the one build runs every W.
#   sim <W>  one launch of W x W x W against an AXI memory holding its
#            operands, every result checked against numpy
#   pnr      placed and routed out of context at 3.333 ns by the recipe the
#            baselines use, movers and control included
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export PYTHONPATH=/scratch/hc676/allo:/scratch/hc676/allo/tests/dataflow/spmw
S=$1; MODE=$2
E=/scratch/hc676/e16_tiles; O=$E/spmw; mkdir -p $O $E/logs
K=$O/kernel_S$S
cd /scratch/hc676/allo
case $MODE in
package)
  J=${3:-4}; LOG=$E/logs/spmw_kernel_S${S}_package.log
  echo "SPMW_TILES_KERNEL_START package S=$S $(date)" > $LOG
  rm -rf $K ${K}_pnr
  T0=$(date +%s)
  timeout 7200 python3 -u scripts/spmw_package_kernel.py --design gemmtiles-256 --size $S \
    --frequency 300 --jobs $J --sim --out $K > $K.package.log 2>&1
  echo "package rc=$? wall_s=$(( $(date +%s) - T0 ))" >> $LOG
  grep -hE "^  HLS:|AXI widths|staged" $K.package.log | head -4 | cut -c1-200 >> $LOG
  echo "SPMW_TILES_KERNEL_DONE $(date)" >> $LOG
  ;;
sim)
  # SIM_BEAT=2 is the bench memory's default, a read beat every other cycle;
  # its record is kept beside the measured one, as W<side>_beat2.
  W=$3; B=${SIM_BEAT:-1}; T=""; [ $B = 1 ] || T=_beat$B
  LOG=$E/logs/spmw_kernel_S${S}_W${W}${T}_sim.log
  echo "SPMW_TILES_KERNEL_START sim S=$S W=$W beat=$B $(date)" > $LOG
  # The simulation goes beside the kernel directory it is given, so each
  # workload gets a name of its own for the one build.
  L=${K}_W$W$T; rm -rf ${L}_sim ${L}_ops; ln -sfn $K $L
  timeout 14400 python3 -u scripts/spmw_array_operands.py --design gemmtiles-$W --size $S \
    --out ${L}_ops > $L.ops.log 2>&1
  echo "operands rc=$?" >> $LOG
  T0=$(date +%s)
  # --beat 1: a memory that returns a beat a cycle within a burst, as the
  # baselines' cosimulation memories do. The model's default, a beat every
  # other cycle, is all of a 512-bit port a 32-lane edge can take.
  python3 -u scripts/spmw_kernel_sim.py $L ${L}_ops --design gemmtiles-$W \
    --size $S --polls 4000000 --beat $B > $L.sim.log 2>&1
  echo "sim rc=$? wall_s=$(( $(date +%s) - T0 ))" >> $LOG
  grep -hE "SPMW TB|xsim:|counts:" $L.sim.log | cut -c1-200 >> $LOG
  rm -f ${L}_sim/*.hex ${L}_ops/*.bin
  echo "SPMW_TILES_KERNEL_DONE $(date)" >> $LOG
  ;;
pnr)
  LOG=$E/logs/spmw_kernel_S${S}_pnr.log
  echo "SPMW_TILES_KERNEL_START pnr S=$S $(date)" > $LOG
  OUT=${K}_pnr; rm -rf $OUT; mkdir -p $OUT; cd $OUT
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
  echo "SPMW_TILES_KERNEL_DONE $(date)" >> $LOG
  ;;
esac
