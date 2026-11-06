#!/bin/bash
# Place and route Allo's library systolic GEMM out of context -- the same
# recipe pnr_matched.sh uses for AutoSA and the SPMW arrays, on the RTL Vitis
# already synthesised. This replaces export_design -flow impl, which hangs
# indefinitely in RTL testbench generation for this design, and it also puts
# Allo on the same measurement flow as every other system in E1 instead of the
# only one that used the Vitis export path.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1
V=/scratch/hc676/e1_allo_beats/S$S/prj/sol/syn/verilog
OUT=/scratch/hc676/e1_allo_ooc/S$S
cd /scratch/hc676 || exit 2                 # driver stays outside the run dir
[ -f "$V/gemm.v" ] || { echo "ALLO_OOC_S${S} NO_RTL at $V"; exit 1; }
command -v vivado >/dev/null || { echo "ALLO_OOC_S${S} ENVIRONMENT vivado not found"; exit 2; }

rm -rf "$OUT"; mkdir -p "$OUT"
cat > "$OUT/pnr.tcl" <<TCL
foreach f [glob -nocomplain $V/*.v]  { read_verilog \$f }
foreach f [glob -nocomplain $V/*.sv] { read_verilog -sv \$f }
synth_design -top gemm -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 3.333 -name ap_clk [get_ports ap_clk]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file util.rpt
report_utilization -hierarchical -file util_hier.rpt
report_timing_summary -file timing.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
TCL
echo "ALLO_OOC_S${S} START $(date -Is)"
T0=$(date +%s)
( cd "$OUT" && exec vivado -mode batch -source pnr.tcl -nojournal -nolog > pnr.log 2>&1 )
rc=$?
echo "ALLO_OOC_S${S} rc=$rc wall_s=$(( $(date +%s) - T0 ))"
grep -E "^PNR_UNROUTED" "$OUT/pnr.log" || echo "  (no PNR_UNROUTED line)"
cd "$OUT" || exit 2
python3 - <<'PY'
import os, re
u = open("util.rpt", errors="replace").read() if os.path.isfile("util.rpt") else ""
t = open("timing.rpt", errors="replace").read() if os.path.isfile("timing.rpt") else ""
def cell(name):
    m = re.search(r"^\|\s*%s\*?\s*\|\s*([\d.]+)\s*\|" % re.escape(name), u, re.M)
    return m.group(1) if m else "-"
m = re.search(r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)", t, re.S)
wns = m.group(1) if m else "-"
print("ALLO_OOC_NUMBERS LUT %s FF %s DSP %s BRAM %s URAM %s WNS %s" %
      (cell("CLB LUTs"), cell("CLB Registers"), cell("DSPs"),
       cell("Block RAM Tile"), cell("URAM"), wns))
PY
[ -f util.rpt ] || { echo "ALLO_OOC_S${S} NO_UTIL -- pnr.log tail:"; tail -15 "$OUT/pnr.log"; }
echo "ALLO_OOC_S${S} END $(date -Is)"
