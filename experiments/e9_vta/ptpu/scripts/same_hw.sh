#!/bin/bash
# same_hw.sh <size>: is the hardware identical across the four programs?
# Compares every role's HLS C++, wrapper and Vitis script, and the fabric.
cd /scratch/hc676/e12_ptpu || exit 2
S=$1
sums() {
  local b=$1
  ( cd "$b" && for r in $(ls -d *_r[0-9]* 2>/dev/null); do
      md5sum "$r/kernel.cpp" "$r/$r.sv" "$r/run.tcl" 2>/dev/null
    done; md5sum spmw_top.sv spmw_fifo.sv spmw_const.sv ) | sort -k2
}
ref=$(sums b_ptpu-micro_$S)
for w in mixed llama dsv4; do
  if [ "$(sums b_ptpu-${w}_$S)" == "$ref" ]; then
    echo "S=$S micro vs $w: identical ($(echo "$ref" | wc -l) files)"
  else
    echo "S=$S micro vs $w: DIFFERENT"; diff <(echo "$ref") <(sums b_ptpu-${w}_$S) | head
  fi
done
