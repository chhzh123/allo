#!/bin/bash
# The matched E1 AutoSA designs: an S x S array computing exactly S*S*S.
# Every difference from the earlier fixed-problem runs is in kernel.h and the
# array partition; kernel.c is the working file verbatim.
set -u
export PATH=/scratch/hc676/allo-agent/bin:/scratch/hc676/autosa/llvm9/bin:$PATH
export LD_LIBRARY_PATH=/scratch/hc676/autosa/llvm9/lib:${LD_LIBRARY_PATH:-}
A=/scratch/hc676/autosa/AutoSA
REF=/scratch/hc676/e1_autosa_2026-09-06/gen/mm8i8z
W=/scratch/hc676/e1_autosa_matched; mkdir -p $W
for S in 4 8 16 32; do
  D=$W/S$S; rm -rf $D; mkdir -p $D/out/src $D/out/latency_est $D/out/resource_est $D/out/tuning
  sed -E "s/#define I [0-9]+/#define I $S/; s/#define J [0-9]+/#define J $S/; s/#define K [0-9]+/#define K $S/" \
      $REF/kernel.h > $D/kernel.h
  cp $REF/kernel.c $D/kernel.c
  cp $REF/simd_info.json $D/ 2>/dev/null
  cd $A
  ./autosa $D/kernel.c --config=./autosa_config/autosa_config.json \
    --target=autosa_hls_c --output-dir=$D/out \
    --sa-sizes="{kernel[]->space_time[3];kernel[]->array_part[$S,$S,$S];kernel[]->latency[1,1];kernel[]->simd[1]}" \
    --simd-info=$D/simd_info.json --hls > $D/gen.log 2>&1
  rc=$?
  k=$D/out/src/kernel_kernel.cpp
  if [ -f "$k" ]; then
    pe=$(grep -cE "^ *PE_wrapper" $k)
    printf "  S=%-3s rc=%s  PE_wrapper instances=%-5s kernel=%s lines\n" "$S" "$rc" "$pe" "$(wc -l < $k)"
    grep -oE "gmem[A-Za-z_]*" $k | sort -u | tr "\n" " " | sed "s/^/       masters: /"; echo
  else
    printf "  S=%-3s rc=%s  no kernel: %s\n" "$S" "$rc" "$(grep -oE 'Error: [^\"]*' $D/gen.log | head -1)"
  fi
done
echo MATCHED_GEN_DONE
