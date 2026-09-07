#!/bin/bash
# Route one design out of context and report what the task's bars need.
# usage: pnr_ooc.sh <rtl-dir> <top> <out-dir>
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
RTL=$1; TOP=$2; OUT=$3
mkdir -p "$OUT"; cd "$OUT" || exit 2
cat > pnr.tcl <<TCL
set files [glob -nocomplain $RTL/*.v $RTL/*.sv]
foreach f \$files { if {[string match *tb_study* \$f]} { continue }
  if {[string match *.sv \$f]} { read_verilog -sv \$f } else { read_verilog \$f } }
synth_design -top $TOP -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 3.333 -name ap_clk [get_ports ap_clk]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file util.rpt
report_timing_summary -file timing.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
TCL
vivado -mode batch -source pnr.tcl -nojournal -nolog > pnr.log 2>&1
grep -E "^PNR_UNROUTED" pnr.log
python3 - <<'PY'
import re
u = open("util.rpt", errors="replace").read() if __import__("os").path.isfile("util.rpt") else ""
t = open("timing.rpt", errors="replace").read() if __import__("os").path.isfile("timing.rpt") else ""
def cell(name, text):
    m = re.search(r"^\|\s*%s\*?\s*\|\s*(\d+)\s*\|" % re.escape(name), text, re.M)
    return int(m.group(1)) if m else None
m = re.search(r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)", t, re.S)
print("PNR LUT", cell("CLB LUTs", u), "FF", cell("CLB Registers", u),
      "DSP", cell("DSPs", u), "WNS", m.group(1) if m else None)
PY
