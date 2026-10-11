#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# fft_build.sh <n> <lanes> [extra args]: the lane-unrolled SPMW FFT at n
# points and <lanes> complex samples a cycle: HLS, RTL cosimulation of 32
# transforms, and place and route at 3.333 ns, into
# /scratch/hc676/e14_fft/spmw_n<n>w<lanes>. PNR= skips place and route.
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export PYTHONPATH=/scratch/hc676/allo:/scratch/hc676/allo/tests/dataflow/spmw
E=/scratch/hc676/e14_fft
export TMPDIR=$E/tmp
mkdir -p "$TMPDIR" $E/logs
N=$1; W=$2; shift 2
OUT=$E/spmw_n${N}w${W}${TAG:-}
LOG=$E/logs/spmw_n${N}w${W}${TAG:-}.log
rm -rf "$OUT"
cd /scratch/hc676/allo || exit 2
T0=$(date +%s)
echo "B_START fftlanes N=$N W=$W $* $(date)" > "$LOG"
timeout 86400 python3 -u scripts/spmw_build_array.py --design fftlanes --size "$N" --lanes "$W" --frequency 300 --out "$OUT" --cosim ${PNR---pnr} "$@" >> "$LOG" 2>&1
echo "B_RC=$? wall_s=$(( $(date +%s) - T0 ))" >> "$LOG"
