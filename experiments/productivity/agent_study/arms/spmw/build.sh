#!/bin/bash
# The SPMW arm, self-contained like the other two: it sets up its own tools
# rather than trusting the caller's environment, so a misconfigured caller
# cannot be scored as a design failure.
# usage: build.sh <trialdir> <vecset>
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
cd /scratch/hc676/allo || exit 2
exec python3 -u /scratch/hc676/agentstudy/arms/spmw/build.py "$1" "${2:-visible}"
