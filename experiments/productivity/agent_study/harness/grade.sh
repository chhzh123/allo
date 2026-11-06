#!/bin/bash
# Grade one submitted trial: held-out vectors, then routing.
# usage: grade.sh <arm> <trial-dir>
set -u
ARM=$1; T=$2; S=/scratch/hc676/agentstudy
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
echo "=== held-out vectors"
case $ARM in
  spmw) (cd /scratch/hc676/allo && python3 -u "$S/arms/spmw/build.py" "$T" held_out) ;;
  hls)  bash "$S/arms/hls/build.sh" "$T" held_out ;;
  rtl)  bash "$S/arms/rtl/build.sh" "$T" held_out ;;
esac
echo "=== routing the submitted design out of context at 3.333 ns"
bash "$S/harness/pnr_ooc.sh" "$T/sim" dut_norm "$T/pnr"
