#!/bin/bash
# The SPMW port's resident-weight cycle rows, re-measured with the integer
# multiply bound to fabric.
#
# The binding is not only a resource directive: a fabric multiplier schedules a
# pipeline stage shallower than the DSP one, so the cell's iteration latency
# drops and, since a partial sum crosses N cells, the startup drops by about
# 2N. The published cycle rows predated the binding while the area rows did
# not, which made those tables internally inconsistent. No `--reuse-build`:
# the point is to rebuild.
#
#   rerun_spmw_bound.sh <workload> <N>      # workload: gemm | conv
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
export PYTHONDONTWRITEBYTECODE=1
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
WL=$1
N=$2
S=/scratch/hc676/e4b_spmw_bound
mkdir -p "$S"
cd /scratch/hc676/e4b_work || exit 2
case "$WL" in
  gemm) ARGS="--workload gemm --gemm 128,128,128" ;;
  conv) ARGS="--workload conv --conv 64,16,16,64" ;;
  *) echo "unknown workload $WL"; exit 2 ;;
esac
name=wl_${WL}_N${N}_x
T0=$(date +%s)
SPMW_BIND_MUL=1 python3 -u e4_spmw_run.py --out "$S/$name" --N "$N" $ARGS \
  --pattern full --seed 0 --resident --jobs 6 > "$S/$name.log" 2>&1
echo "$name rc=$? wall=$(( $(date +%s) - T0 ))s" >> "$S/$name.log"
touch "$S/$name.done"
