#!/bin/bash
# One SPMW microbenchmark point: HLS the roles, assemble, and either cosimulate
# for cycles or route for area and timing. Detached; the caller polls the marker.
#   run_spmw_micro.sh <size> <cosim|pnr> [design] [jobs]
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
export PYTHONDONTWRITEBYTECODE=1
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1
MODE=$2
DESIGN=${3:-tpumicro}
JOBS=${4:-8}
TAG=${DESIGN#tpumicro}; TAG=${TAG#-}
NAME="spmw_${MODE}${TAG:+_$TAG}_S${S}"
OUT=/scratch/hc676/e3_micro/$NAME
LOG=/scratch/hc676/e3_micro/logs/$NAME.log
mkdir -p /scratch/hc676/e3_micro/logs
rm -f "$LOG.done"
FLAG="--cosim"; [ "$MODE" = "pnr" ] && FLAG="--pnr"
cd /scratch/hc676/e3_micro/allo || exit 2
T0=$(date +%s)
python3 -u scripts/spmw_build_array.py --design "$DESIGN" --size "$S" \
  --frequency 300 --jobs "$JOBS" $FLAG --out "$OUT" > "$LOG" 2>&1
echo "rc=$? wall_s=$(( $(date +%s) - T0 ))" >> "$LOG"
touch "$LOG.done"
