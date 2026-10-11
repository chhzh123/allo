#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# probe32.sh: does the FIFO-weight variant of the Allo array get through csynth at 32x32?
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
cd /scratch/hc676/e15_ws
rm -rf probe/S32; mkdir -p probe
PYTHONPATH=/scratch/hc676/allo python3 ws_probe2.py 32 probe/S32 > probe/S32.gen.log 2>&1
cd probe/S32 || exit 1
cp ../S16/csynth.tcl .
T0=$(date +%s)
timeout 12600 vitis_hls -f csynth.tcl > csynth.log 2>&1
echo "PROBE32 rc=$? wall_s=$(( $(date +%s) - T0 ))" > ../S32.done
