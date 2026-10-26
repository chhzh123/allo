#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# e9_simtime_controlled.sh <tree> <out> [phase...]: the mini-TPU's simulation
# times remeasured on a quiet machine, interleaved (e9_simtime_controlled.py
# says how). <tree> is a checkout of this branch whose allo/_mlir links a
# build of the same C++. Run it detached; it waits for the machine to be quiet
# before every run, so it can take far longer than the runs themselves.
main() {
set -u
local tree=$1 out=$2
shift 2
export PATH=/scratch/hc676/allo-agent/bin:$PATH
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
export PYTHONPATH=$tree:$tree/tests/dataflow/spmw
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
# The generated top copies each operand into a local array: megabytes for
# these programs, so Vitis's C simulation needs the stack unbounded.
ulimit -s unlimited
mkdir -p "$out"
exec python3 -u "$tree/experiments/e9_vta/scripts/e9_simtime_controlled.py" "$tree" "$out" "$@"
}
main "$@"
