#!/bin/bash
# run_rocc_xsim.sh <variant> <dim> <stim> <tag>: one stimulus on the whole
# Gemmini accelerator, driven by RoCC commands (gemmini_rocc_bench.py).
# Registers and memories start at zero, as an FPGA's do after configuration.
# PROGRAM=batched runs the hand-scheduled program instead of the library's;
# BUS=128 runs the build with the 128-bit system bus.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export PATH=/scratch/hc676/allo-agent/bin:$PATH
ROOT=/scratch/hc676/gemmini_full
VAR=$1; D=$2; STIM=$3; TAG=$4
V=$ROOT/out/${VAR}_${D}${BUS:+_b$BUS}_sim
W=$ROOT/sim/${TAG}_${VAR}_$D${BUS:+_b$BUS}
LOG=$W.log
# The matmul-only configuration has the shift variant's arithmetic.
BV=$VAR; [ "$VAR" = matmul ] && BV=shift
rm -rf "$W"; mkdir -p "$W"
python3 $ROOT/gemmini_rocc_bench.py --rtl "$V" --stim "$STIM" --out "$W" --variant "$BV" \
  --program "${PROGRAM:-library}" > "$LOG" 2>&1 || exit 1
python3 $ROOT/closure.py "$V" Gemmini 2>/dev/null | sed "s|^|$V/|" > "$W/files.f"
cd "$W" || exit 2
T0=$(date +%s)
{ xvlog -sv -d RANDOMIZE_REG_INIT -d RANDOMIZE_MEM_INIT -d "RANDOM=32'h0" -f files.f tb.sv \
    && xelab tb -s tbsim -timescale 1ns/1ps \
    && xsim tbsim -runall; } >> "$LOG" 2>&1
echo "rc=$? wall_s=$(( $(date +%s) - T0 ))" >> "$LOG"
# The waveform database and the compiled snapshot are the bulk of a run.
rm -rf xsim.dir tbsim.wdb mem.hex
