#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_tapa_rs_pnr.sh <W>: autosa_tapa_pnr.sh's out-of-context Vivado flow (same steps, same
# 3.333 ns clock, same report parser) on RapidStream's RTL, plus its floorplan.xdc
# applied after synthesis, as Vitis applies it to a linked kernel. Reports each
# pblock's cells so an empty floorplan cannot pass unnoticed. Timing is reported
# first and the routed design saved, so a failing report cannot lose them.
main() {
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
local W=$1 E=/scratch/hc676/e22_autosa_tapa/e12rs
local S=$E/rs/W$W/dse/solution_0 P=$E/pnr/W$W
local R=$S/hdl
[ -f $R/kernel0.v ] || { echo "W=$W pnr: no RTL in $R" >> $E/pnr_results.txt; return 1; }
rm -rf $P; mkdir -p $P; cd $P || return 1
cp $S/xdc/floorplan.xdc .
cat > pnr.tcl <<TCL
foreach f [glob -nocomplain $R/*.v] { read_verilog \$f }
foreach f [glob -nocomplain $R/*.sv] { read_verilog -sv \$f }
synth_design -top kernel0 -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 3.333 -name ap_clk [get_ports ap_clk]
read_xdc $P/floorplan.xdc
foreach pb [get_pblocks] {
  puts "PBLOCK \$pb IS_SOFT [get_property IS_SOFT \$pb] top_cells [llength [get_cells -quiet -of_objects \$pb]] leaf_cells [llength [get_cells -quiet -hierarchical -filter "PBLOCK == \$pb && IS_PRIMITIVE"]]"
}
opt_design
place_design
phys_opt_design
route_design
report_timing_summary -file timing.rpt
report_route_status -file route_status.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
write_checkpoint -force routed.dcp
report_utilization -file util.rpt
report_utilization -hierarchical -hierarchical_depth 1 -file util_hier.rpt
foreach pb [get_pblocks] { report_utilization -pblocks \$pb -append -file util_pblocks.rpt }
TCL
local t0=$(date +%s)
vivado -mode batch -source pnr.tcl -nojournal -log pnr.log > /dev/null 2>&1
local rc=$?
local line=$(python3 - <<'PY'
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
)
echo "W=$W rc=$rc wall_s=$(( $(date +%s) - t0 )) $(grep -o 'PNR_UNROUTED [0-9]*' pnr.log) $line" >> $E/pnr_results.txt
rm -rf .Xil
}
main "$@"
