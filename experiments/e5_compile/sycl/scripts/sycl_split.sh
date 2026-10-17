#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# sycl_split.sh <S>...: what the SYCL compile does before Intel's HLS backend
# starts, step by step: the same command as sycl_compile.sh with -v, its
# echoed sub-commands stamped, stopped as soon as aoc begins (the campaign
# times aoc itself). The host compile is the step Vitis csynth has no
# counterpart for: it compiles the host program, not the kernels.
main() {
local args=("$@")
source /scratch/hc676/intel/env.sh
set -u
set -- "${args[@]}"
E=/scratch/hc676/e17_sycl
for s in "$@"; do
  D=$E/split/S$s
  rm -rf $D; mkdir -p $D; cd $D || exit 1
  setsid bash -c "icpx -v -fsycl -fintelfpga -DSIDE=$s -Xshardware -fsycl-link=early \
    -Xstarget=Agilex7 -Xsclock=300MHz -Xsv $E/src/e1_sycl.cpp -o e1.a 2>&1 | python3 -u $E/stamp.py > build.log" &
  local pg=$!
  # -v quotes only a command's program; aoc is an argument of llvm-foreach.
  until grep -q "llvm-foreach.*oclfpga/bin/aoc " build.log 2>/dev/null || ! kill -0 $pg 2>/dev/null; do sleep 1; done
  kill -- -$pg 2>/dev/null
  sleep 2
  python3 $E/split_row.py build.log > split.txt
  echo "S=$s $(tr '\n' ' ' < split.txt)"
done
}
main "$@"
