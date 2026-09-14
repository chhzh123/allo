#!/bin/bash
# Does binding the multiply to fabric change the port's *cycles*?
#
# It changes the schedule, not just the resource: E3's cell went from an
# iteration latency of 5 to 4, because a fabric multiplier is scheduled
# shallower than the DSP one at these widths. A partial sum crosses N cells, so
# a shallower cell should show up as a shorter startup and leave the interval
# alone. This runs the same point twice, binding on and off, on a few tiles --
# startup is a first-output number and does not need the whole workload.
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
export PYTHONDONTWRITEBYTECODE=1
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
N=${1:-8}
W=/scratch/hc676/e4_probe
mkdir -p "$W"
cd /scratch/hc676/e4b_work || exit 2
for bind in ${BINDS:-1 0}; do
  out=$W/N${N}_bind${bind}
  rm -rf "$out"
  echo "### N=$N SPMW_BIND_MUL=$bind  $(date +%T)"
  SPMW_BIND_MUL=$bind python3 -u e4_spmw_run.py --N "$N" --workload gemm \
    --gemm 128,128,128 --pattern small --seed 0 --resident --limit 8 \
    --jobs 6 --out "$out" 2>&1 | tail -2
done
echo "PROBE DONE $(date +%T)"
