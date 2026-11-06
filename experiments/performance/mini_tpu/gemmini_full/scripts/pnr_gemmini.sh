#!/bin/bash
# pnr_gemmini.sh <variant> <dim> [period [tag]]: route Gemmini's whole accelerator
# out of context. The recipe is the one VTA's Core was routed with: 3.333 ns,
# xcu280, retiming in synthesis and in both physical optimisations. Gemmini's
# pipelined scale units put their registers after the logic and leave the
# balancing to retiming, so it needs this more than VTA does.
# The one change to Gemmini's RTL is ramfix.py's, to its widest memories.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
ROOT=/scratch/hc676/gemmini_full
export TMPDIR=$ROOT/tmp; mkdir -p "$TMPDIR"
VAR=$1; D=$2; PERIOD=${3:-3.333}; TAG=${4:-}
V=$ROOT/out/${VAR}_$D
OUT=$ROOT/pnr_${VAR}_$D${TAG:+_$TAG}
[ -f "$V/Gemmini.sv" ] || { echo "NO_RTL in $V"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
python3 $ROOT/closure.py "$V" Gemmini > "$OUT/files.txt" 2> "$OUT/closure.txt"
python3 $ROOT/ramfix.py "$V" "$OUT/ramfix" > "$OUT/ramfix.txt"
cat > "$OUT/pnr.tcl" <<TCL
set fh [open files.txt]
foreach f [split [string trim [read \$fh]] "\n"] {
  if {[file exists ramfix/\$f]} { read_verilog -sv ramfix/\$f } else { read_verilog -sv $V/\$f }
}
close \$fh
synth_design -top Gemmini -part xcu280-fsvh2892-2L-e -mode out_of_context -retiming -verilog_define SYNTHESIS
create_clock -period $PERIOD -name clk [get_ports clock]
opt_design
place_design
phys_opt_design -retime
route_design
phys_opt_design -retime
report_utilization -file util.rpt
report_utilization -hierarchical -hierarchical_depth 6 -file util_hier.rpt
report_timing_summary -file timing.rpt
puts "PNR_UNROUTED [llength [get_nets -filter {ROUTE_STATUS == UNROUTED} -quiet]]"
write_checkpoint -force routed.dcp
TCL
T0=$(date +%s)
( cd "$OUT" && exec vivado -mode batch -source pnr.tcl -nojournal -nolog > pnr.log 2>&1 )
echo "rc=$? wall_s=$(( $(date +%s) - T0 ))" > "$OUT/DONE"
grep -E "^PNR_UNROUTED" "$OUT/pnr.log" >> "$OUT/DONE" 2>/dev/null
