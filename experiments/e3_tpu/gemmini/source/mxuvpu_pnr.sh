#!/bin/bash
# Elaborate the Gemmini MXU+VPU at one size and route it out of context, the
# same recipe E1 uses for every other design: 3.333 ns, xcu280, no shell.
# Differencing against E1's mesh-only Gemmini rows gives the VPU's cost.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
source /scratch/hc676/gemmini_env.sh
S=$1
B=/scratch/hc676/gemmini_mxuvpu
OUT=/scratch/hc676/gemmini_mxuvpu_pnr/S$S
cd /scratch/hc676 || exit 2

echo "MXUVPU_S${S} ELABORATE $(date -Is)"
( cd "$B" && MESH_DIM=$S timeout 7200 sbt -batch "runMain gen.ElaborateMxuVpu" ) > /scratch/hc676/mxuvpu_elab_S$S.log 2>&1
grep -E "MXUVPU_ELABORATE_OK" /scratch/hc676/mxuvpu_elab_S$S.log || {
  echo "MXUVPU_S${S} ELABORATE_FAILED"; tail -8 /scratch/hc676/mxuvpu_elab_S$S.log; exit 1; }

V=$B/mxuvpu_out_$S
[ -f "$V/MxuVpu.v" ] || { echo "MXUVPU_S${S} NO_RTL in $V"; ls "$V" | head; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"
cat > "$OUT/pnr.tcl" <<TCL
foreach f [glob -nocomplain $V/*.v]  { read_verilog \$f }
foreach f [glob -nocomplain $V/*.sv] { read_verilog -sv \$f }
synth_design -top MxuVpu -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 3.333 -name clk [get_ports clock]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file util.rpt
report_timing_summary -file timing.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
TCL
echo "MXUVPU_S${S} PNR_START $(date -Is)"
T0=$(date +%s)
( cd "$OUT" && exec vivado -mode batch -source pnr.tcl -nojournal -nolog > pnr.log 2>&1 )
echo "MXUVPU_S${S} rc=$? wall_s=$(( $(date +%s) - T0 ))"
grep -E "^PNR_UNROUTED" "$OUT/pnr.log" || echo "  (no PNR_UNROUTED line)"
cd "$OUT" || exit 2
python3 - <<'PY'
import os, re
u = open("util.rpt", errors="replace").read() if os.path.isfile("util.rpt") else ""
t = open("timing.rpt", errors="replace").read() if os.path.isfile("timing.rpt") else ""
def cell(n):
    m = re.search(r"^\|\s*%s\*?\s*\|\s*([\d.]+)\s*\|" % re.escape(n), u, re.M)
    return m.group(1) if m else "-"
m = re.search(r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)", t, re.S)
print("MXUVPU_NUMBERS LUT %s FF %s DSP %s BRAM %s WNS %s" %
      (cell("CLB LUTs"), cell("CLB Registers"), cell("DSPs"),
       cell("Block RAM Tile"), m.group(1) if m else "-"))
PY
[ -f util.rpt ] || { echo "MXUVPU_S${S} NO_UTIL"; tail -12 "$OUT/pnr.log"; }
echo "MXUVPU_S${S} END $(date -Is)"
