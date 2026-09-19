#!/bin/bash
# The Gemmini transformer-block engine, routed out of context on the U280.
#
# Not at 3.333 ns: E3 measured the multiply scale form at WNS -1.373 ns at 4x4,
# and this design is that form plus a hardfloat divider, square root and
# reciprocal. The period is swept so the achieved clock is measured rather than
# assumed -- time to finish is cycles x period, so the clock is part of the
# answer, not a constraint to satisfy.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
PERIOD=$1
TAG=${2:-}
V=/scratch/hc676/e3_micro/gemmini/mxuvpunorm_out_16$TAG
OUT=/scratch/hc676/e8_block/gem_pnr_${PERIOD/./p}${TAG}
cd /scratch/hc676 || exit 2
[ -f "$V/MxuVpuNorm.v" ] || { echo "NO_RTL in $V"; ls "$V" 2>/dev/null | head; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
cat > "$OUT/pnr.tcl" <<TCL
foreach f [glob -nocomplain $V/*.v]  { read_verilog \$f }
foreach f [glob -nocomplain $V/*.sv] { read_verilog -sv \$f }
# `-retiming`: `AccumulatorScale`'s activation and float scale are one
# combinational cloud with a `Pipe` on its output, so the registers that
# break it have to be moved backwards into it.  Without this the design
# routes at 29.7 ns whatever the constraint.
synth_design -top MxuVpuNorm -part xcu280-fsvh2892-2L-e -mode out_of_context -retiming
create_clock -period $PERIOD -name clk [get_ports clock]
opt_design
place_design
phys_opt_design -retime
route_design
phys_opt_design -retime
report_utilization -file util.rpt
report_timing_summary -file timing.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
TCL
T0=$(date +%s)
( cd "$OUT" && exec vivado -mode batch -source pnr.tcl -nojournal -nolog > pnr.log 2>&1 )
echo "rc=$? wall_s=$(( $(date +%s) - T0 ))" > "$OUT/DONE"
grep -E "^PNR_UNROUTED" "$OUT/pnr.log" >> "$OUT/DONE" 2>/dev/null
