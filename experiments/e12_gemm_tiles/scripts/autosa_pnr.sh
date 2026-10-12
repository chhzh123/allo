#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_pnr.sh <variant> <S> <W>: place and route an AutoSA build that
# autosa_tiles2.sh or autosa_tiles3.sh made with PNR=0, from the RTL its
# synthesis left: out of context at 3.333 ns, the recipe of every other row.
# <variant> is the build's directory under autosa/, e.g. wide_k32_l1x4.
main() {
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
V=$1; S=$2; W=$3
E=/scratch/hc676/e16_tiles
D=$E/autosa/$V/S${S}_W$W
LOG=$E/logs/autosa_${V}_S${S}_W$W.log
VV=$D/hls/cosim_prj/sol/syn/verilog
[ -f $VV/kernel0.v ] || { echo "pnr: no RTL in $VV" >> $LOG; exit 1; }
OUT=$D/pnr
rm -rf $OUT; mkdir -p $OUT; cd $OUT || exit 1
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
sed -i '/^pnr: not run$/d' $LOG
cat $D/pnr_lines.txt >> $LOG
echo "AUTOSA_PNR_DONE $(date)" >> $LOG
}
main "$@"
