#!/bin/bash
# run_acc_xsim.sh <dim> [variant [stim [root]]]: a stimulus on Gemmini MxuAccVpu
# at mesh side <dim>; by default E3's S=16 one on E3's configuration. The
# stimulus file's header gives the tile's shape.
# Registers and memories start at zero, as an FPGA's do after configuration:
# Gemmini's mesh registers have no reset, and their X would otherwise poison
# the accumulator's read-side feedback registers in simulation.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export PATH=/scratch/hc676/allo-agent/bin:$PATH
D=$1
VAR=${2:-}
STIM=${3:-/scratch/hc676/e9_lean/vta_bench/stim/stim_S16.txt}
ROOT=${4:-/scratch/hc676/e9_lean/gem}
V=/scratch/hc676/e3_micro/gemmini_elab/mxuaccvpu_out_${D}${VAR:+_$VAR}
W=$ROOT/xsim_d${D}
LOG=$ROOT/xsim_d${D}.log
rm -rf "$W"; mkdir -p "$W"
cp "$V/MxuAccVpu.v" "$W/"
python3 /scratch/hc676/e9_lean/gem/gen_mxuaccvpu_tb.py --dim "$D" \
  --stim "$STIM" --out "$W/tb.sv" > "$LOG" 2>&1
cd "$W" || exit 2
T0=$(date +%s)
ZERO="RANDOM=32'h0"
{ xvlog -d RANDOMIZE_REG_INIT -d RANDOMIZE_MEM_INIT -d "$ZERO" MxuAccVpu.v \
    && xvlog -sv tb.sv && xelab tb -s tbsim -timescale 1ns/1ps \
    && xsim tbsim -runall; } >> "$LOG" 2>&1
echo "rc=$? wall_s=$(( $(date +%s) - T0 ))" >> "$LOG"
