#!/bin/bash
# build.sh <design> <size> [extra args]: one LLaMA array build, cosim + P&R.
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export PYTHONPATH=/scratch/hc676/allo:/scratch/hc676/allo/tests/dataflow/spmw
export TMPDIR=/scratch/hc676/e10_llama/tmp
mkdir -p "$TMPDIR" /scratch/hc676/e10_llama/logs
D=$1; S=$2; shift 2
TAG=${TAG:-}
OUT=/scratch/hc676/e10_llama/b_${D}_${S}${TAG}
LOG=/scratch/hc676/e10_llama/logs/b_${D}_${S}${TAG}.log
rm -rf "$OUT"
cd /scratch/hc676/allo || exit 2
T0=$(date +%s)
echo "B_START $D S=$S $* $(date)" > "$LOG"
timeout 72000 python3 -u scripts/spmw_build_array.py --design "$D" --size "$S" --out "$OUT" --cosim --pnr "$@" >> "$LOG" 2>&1
echo "B_RC=$? wall_s=$(( $(date +%s) - T0 ))" >> "$LOG"
