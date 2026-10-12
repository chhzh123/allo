#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_gen.sh <S> <W> <KT> <L> [autosa flags...]: generate, and only
# generate, AutoSA's S x S array for a W x W x W matmul with
# array_part[S*L, S*L, KT], latency[L, L], simd[1] and --host-serialize, to
# read what the generated design moves a beat at each level of its I/O
# network before spending an hour synthesising it. Seconds a design.
main() {
set -u
export PATH=/scratch/hc676/allo-agent/bin:/scratch/hc676/autosa/llvm9/bin:$PATH
export LD_LIBRARY_PATH=/scratch/hc676/autosa/llvm9/lib:${LD_LIBRARY_PATH:-}
S=$1; W=$2; KT=$3; L=$4; shift 4
A=/scratch/hc676/autosa/AutoSA
REF=/scratch/hc676/e1_autosa_2026-09-06/gen/mm8i8z
TAG=S${S}_W${W}_k${KT}_l${L}$(echo "$*" | tr -c 'A-Za-z0-9\n' '_' | cut -c1-60)
D=/scratch/hc676/e16_tiles/autosa_gen/$TAG
rm -rf $D; mkdir -p $D/out/src $D/out/latency_est $D/out/resource_est $D/out/tuning
sed -E "s/#define I [0-9]+/#define I $W/; s/#define J [0-9]+/#define J $W/; s/#define K [0-9]+/#define K $W/" $REF/kernel.h > $D/kernel.h
sed -E "s/\(data_t\)\(rand\(\) % 8\)/(data_t)(rand() % 256 - 128)/" $REF/kernel.c > $D/kernel.c
cp $REF/simd_info.json $D/
cd $A
./autosa $D/kernel.c --config=./autosa_config/autosa_config.json \
  --target=autosa_hls_c --output-dir=$D/out --host-serialize "$@" \
  --sa-sizes="{kernel[]->space_time[3];kernel[]->array_part[$((S*L)),$((S*L)),$KT];kernel[]->latency[$L,$L];kernel[]->simd[1]}" \
  --simd-info=$D/simd_info.json --hls > $D/gen.log 2>&1
rc=$?
G=$D/out/src/kernel_kernel.cpp
echo "== $TAG gen rc=$rc"
[ -f $G ] || { grep -oE "Error: .*|error: .*" $D/gen.log | head -3; return 1; }
echo "  ports: $(grep -oE 'void kernel0\([^)]*' $G | head -1 | sed 's/void kernel0(//') | cells $(grep -cE '^ *PE_wrapper' $G)"
for f in A_IO_L2_in B_IO_L2_in PE C_drain_IO_L1_out C_drain_IO_L2_out C_drain_IO_L3_out; do
  echo "  $f: $(grep -E "^void $f\(" $G | grep -oE 'hls::stream<[A-Za-z0-9_ ]+> &[A-Za-z_]+' | sed -E 's/hls::stream<([^>]+)> &fifo_/\1 /' | tr '\n' ',' | sed 's/,$//')"
done
echo "  local buffers: $(grep -oE '(A_t[0-9]+|B_t[0-9]+|C_t[0-9]+|char|int) local_[ABC](_ping)?\[[0-9]+\]\[[0-9]+\]' $G | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
}
main "$@"
