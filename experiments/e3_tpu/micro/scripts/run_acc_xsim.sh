#!/bin/bash
# run_acc_xsim.sh <dim>: the S=16 stimulus on Gemmini MxuAccVpu at mesh side <dim>.
# Registers and memories start at zero, as an FPGA's do after configuration:
# Gemmini's mesh registers have no reset, and their X would otherwise poison
# the accumulator's read-side feedback registers in simulation.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export PATH=/scratch/hc676/allo-agent/bin:$PATH
D=$1
V=/scratch/hc676/e3_micro/gemmini_elab/mxuaccvpu_out_${D}
W=/scratch/hc676/e9_lean/gem/xsim_d${D}
LOG=/scratch/hc676/e9_lean/gem/xsim_d${D}.log
rm -rf "$W"; mkdir -p "$W"
cp "$V/MxuAccVpu.v" "$W/"
python3 /scratch/hc676/e9_lean/gem/gen_mxuaccvpu_tb.py --dim "$D" --tiles 16 --size 16 \
  --stim /scratch/hc676/e9_lean/vta_bench/stim/stim_S16.txt --out "$W/tb.sv" > "$LOG" 2>&1
cd "$W" || exit 2
T0=$(date +%s)
ZERO="RANDOM=32'h0"
{ xvlog -d RANDOMIZE_REG_INIT -d RANDOMIZE_MEM_INIT -d "$ZERO" MxuAccVpu.v \
    && xvlog -sv tb.sv && xelab tb -s tbsim -timescale 1ns/1ps \
    && xsim tbsim -runall; } >> "$LOG" 2>&1
echo "rc=$? wall_s=$(( $(date +%s) - T0 ))" >> "$LOG"
