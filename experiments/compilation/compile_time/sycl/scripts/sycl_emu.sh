#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# sycl_emu.sh <S>...: build E1's SYCL array for Intel's FPGA emulator and run it
# against its own reference, a size at a time. Every cell is a kernel that
# blocks on its pipes, so the emulator is given a worker thread a kernel
# (S^2 + 3); the run is pinned to eight cores and bounded at ten minutes.
main() {
# oneAPI's vars.sh reads variables it has not set and resets the positional
# parameters of whatever sources it: keep the arguments, source it, then set -u.
local args=("$@")
source /scratch/hc676/intel/env.sh
set -u
set -- "${args[@]}"
E=/scratch/hc676/e17_sycl
mkdir -p $E/emu $E/logs; cd $E/emu
for s in "$@"; do
  L=$E/logs/emu_S$s.log
  echo "EMU_START S=$s $(date -Iseconds)" > $L
  t0=$(date +%s)
  icpx -fsycl -fintelfpga -DFPGA_EMULATOR -DSIDE=$s $E/src/e1_sycl.cpp -o emu_S$s > emu_S$s.build.log 2>&1
  echo "build rc=$? wall_s=$(( $(date +%s) - t0 ))" >> $L
  [ -x emu_S$s ] || continue
  t0=$(date +%s)
  CL_CONFIG_CPU_TBB_NUM_WORKERS=$(( s * s + 16 )) timeout 600 taskset -c 0-7 ./emu_S$s >> $L 2>&1
  echo "run rc=$? wall_s=$(( $(date +%s) - t0 ))" >> $L
  echo "EMU_DONE $(date -Iseconds)" >> $L
done
}
main "$@"
