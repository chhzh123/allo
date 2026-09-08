#!/bin/bash
# Place and route one matched AutoSA design out of context, same recipe as the
# SPMW arrays: 3.333 ns, no shell.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1
V=/scratch/hc676/e1_autosa_matched/hls/S$S/cosim_prj/sol/syn/verilog
OUT=/scratch/hc676/e1_autosa_matched/pnr/S$S
[ -d "$V" ] || { echo "S=$S no RTL"; exit 1; }
rm -rf $OUT; mkdir -p $OUT; cd $OUT
cat > pnr.tcl <<TCL
foreach f [glob -nocomplain $V/*.v] { read_verilog \$f }
foreach f [glob -nocomplain $V/*.sv] { read_verilog -sv \$f }
synth_design -top kernel0 -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 3.333 -name ap_clk [get_ports ap_clk]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file util.rpt
report_timing_summary -file timing.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
TCL
T0=$(date +%s)
vivado -mode batch -source pnr.tcl -nojournal -nolog > pnr.log 2>&1
echo "S=$S rc=$? wall_s=$(( $(date +%s) - T0 ))"
grep -E "^PNR_UNROUTED" pnr.log
python3 - <<'PY'
import os, re
u = open("util.rpt", errors="replace").read() if os.path.isfile("util.rpt") else ""
t = open("timing.rpt", errors="replace").read() if os.path.isfile("timing.rpt") else ""
def cell(name):
    m = re.search(r"^\|\s*%s\*?\s*\|\s*([\d.]+)\s*\|" % re.escape(name), u, re.M)
    return m.group(1) if m else "-"
m = re.search(r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)", t, re.S)
print("  LUT %s  FF %s  DSP %s  BRAM %s  WNS %s" %
      (cell("CLB LUTs"), cell("CLB Registers"), cell("DSPs"),
       cell("Block RAM Tile"), m.group(1) if m else "-"))
PY
echo "PNR_MATCHED_DONE_$S"
