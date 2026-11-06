#!/bin/bash
# The matched AutoSA designs again, this time fetching 512-bit beats like SPMW.
# --host-serialize lays the operands out in the order the array consumes them,
# which is what lets data_pack reach its 64-byte bound.
set -u
export PATH=/scratch/hc676/allo-agent/bin:/scratch/hc676/autosa/llvm9/bin:$PATH
export LD_LIBRARY_PATH=/scratch/hc676/autosa/llvm9/lib:${LD_LIBRARY_PATH:-}
A=/scratch/hc676/autosa/AutoSA
REF=/scratch/hc676/e1_autosa_2026-09-06/gen/mm8i8z
W=/scratch/hc676/e1_autosa_wide
for S in 4 8 16 32; do
  D=$W/S$S; rm -rf $D
  mkdir -p $D/out/src $D/out/latency_est $D/out/resource_est $D/out/tuning
  sed -E "s/#define I [0-9]+/#define I $S/; s/#define J [0-9]+/#define J $S/; s/#define K [0-9]+/#define K $S/" \
      $REF/kernel.h > $D/kernel.h
  cp $REF/kernel.c $D/kernel.c; cp $REF/simd_info.json $D/
  cd $A
  ./autosa $D/kernel.c --config=./autosa_config/autosa_config.json \
    --target=autosa_hls_c --output-dir=$D/out --host-serialize \
    --sa-sizes="{kernel[]->space_time[3];kernel[]->array_part[$S,$S,$S];kernel[]->latency[1,1];kernel[]->simd[1]}" \
    --simd-info=$D/simd_info.json --hls > $D/gen.log 2>&1
  k=$D/out/src/kernel_kernel.cpp
  printf "S=%-3s " $S
  if [ -f "$k" ]; then
    printf "ports: %s | " "$(grep -oE 'void kernel0\([^)]*' $k | head -1 | sed 's/void kernel0(//')"
    printf "PEs: %s | " "$(grep -cE '^void PE_wrapper' $k)"
    printf "drains: %s\n" "$(grep -cE '^void C_drain_IO_L1_out(_wrapper)?_?[a-z]*\(' $k)"
  else
    grep -oE "Error: .*" $D/gen.log | head -1
  fi
done
echo GEN_WIDE_DONE
