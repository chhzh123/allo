#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# same_hw.sh <size>: is the hardware the same whatever the program?
# Compares every role's HLS C++, wrapper and Vitis script, and the fabric,
# between the microbenchmark's build and the five-GEMM program's.
cd /scratch/hc676/e13_mem || exit 2
S=$1
sums() {
  local b=$1
  ( cd "$b" && for r in $(ls -d *_r[0-9]* 2>/dev/null); do
      md5sum "$r/kernel.cpp" "$r/$r.sv" "$r/run.tcl" 2>/dev/null
    done; md5sum spmw_top.sv spmw_fifo.sv spmw_const.sv ) | sort -k2
}
ref=$(sums b_ptpumem-micro_$S)
for w in mixed; do
  if [ "$(sums b_ptpumem-${w}_$S)" == "$ref" ]; then
    echo "S=$S micro vs $w: identical ($(echo "$ref" | wc -l) files)"
  else
    echo "S=$S micro vs $w: DIFFERENT"; diff <(echo "$ref") <(sums b_ptpumem-${w}_$S) | head
  fi
done
