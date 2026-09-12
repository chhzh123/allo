#!/bin/bash
# The Gemmini streaming microbenchmark under xsim, on the emitted Verilog.
# Same protocol and same stimulus file as MxuVpuStream.scala; validated against
# it at S=4 before being trusted at 8 and 16.
#
# Chisel emits no `timescale, so xelab is given one rather than the RTL edited.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export PATH=/scratch/hc676/allo-agent/bin:$PATH
S=$1
V=/scratch/hc676/e3_micro/gemmini_elab/mxuvpu_out_${S}_shift
W=/scratch/hc676/e3_micro/xsim_S${S}
LOG=/scratch/hc676/e3_micro/logs/gem_xsim_S${S}.log
mkdir -p /scratch/hc676/e3_micro/logs
rm -f "$LOG.done"
[ -f "$V/MxuVpu.v" ] || { echo "no RTL at $V" > "$LOG"; touch "$LOG.done"; exit 1; }
rm -rf "$W"; mkdir -p "$W"
cp "$V/MxuVpu.v" "$W/"
python3 /scratch/hc676/e3_micro/gen_mxuvpu_tb.py --size "$S" --tiles 16 \
  --stim /scratch/hc676/e3_micro/stim/stim_S${S}.txt --out "$W/tb.sv" > "$LOG" 2>&1
cd "$W" || exit 2
T0=$(date +%s)
{ xvlog MxuVpu.v && xvlog -sv tb.sv && xelab tb -s tbsim -timescale 1ns/1ps \
    && xsim tbsim -runall; } >> "$LOG" 2>&1
echo "rc=$? wall_s=$(( $(date +%s) - T0 ))" >> "$LOG"
touch "$LOG.done"
