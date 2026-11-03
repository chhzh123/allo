#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# fe_run.sh: time the frontends Fig. 13(b)'s AutoSA (Vitis HLS) and Allo lines
# leave out -- AutoSA's --target=autosa_hls_c code generation and Allo's
# customize + schedule + build(vitis_hls) -- on E1's own inputs, one run per
# size, one at a time. Each run gets a fresh directory and its output is
# compared byte for byte with the file E1 synthesized.
set -u
W=/scratch/hc676/e22_autosa_tapa/fe
OUT=$W/runs
LOG=$W/fe.ndjson
mkdir -p $OUT
now() { date +%s.%N; }
load() { cut -d' ' -f1-3 /proc/loadavg; }
dt() { awk -v a=$1 -v b=$2 'BEGIN{printf "%.2f", b-a}'; }

autosa_one() {  # S E1_DIR EXTRA
  local S=$1 SRC=$2 EXTRA=$3 D=$OUT/autosa_S$1
  rm -rf $D; mkdir -p $D/out/src $D/out/latency_est $D/out/resource_est $D/out/tuning
  cp $SRC/kernel.c $SRC/kernel.h $SRC/simd_info.json $D/
  (
    export PATH=/scratch/hc676/allo-agent/bin:/scratch/hc676/autosa/llvm9/bin:$PATH
    export LD_LIBRARY_PATH=/scratch/hc676/autosa/llvm9/lib:${LD_LIBRARY_PATH:-}
    cd /scratch/hc676/autosa/AutoSA
    local L0=$(load) T0=$(now) START=$(date +%FT%T)
    ./autosa $D/kernel.c --config=./autosa_config/autosa_config.json --target=autosa_hls_c \
      --output-dir=$D/out $EXTRA \
      --sa-sizes="{kernel[]->space_time[3];kernel[]->array_part[$S,$S,$S];kernel[]->latency[1,1];kernel[]->simd[1]}" \
      --simd-info=$D/simd_info.json --hls > $D/gen.log 2>&1
    local rc=$? T1=$(now) L1=$(load)
    local same_kernel=false same_cmd=false
    cmp -s $D/out/src/kernel_kernel.cpp $SRC/out/src/kernel_kernel.cpp && same_kernel=true
    cmp -s <(sed "s|$D|E1|g" $D/out/src/cmd) <(sed "s|$SRC|E1|g" $SRC/out/src/cmd) && same_cmd=true
    printf '{"flow":"autosa_hls_c","size":%d,"rc":%d,"start":"%s","frontend_s":%s,"same_kernel_as_e1":%s,"same_cmd_as_e1":%s,"e1_dir":"%s","load_start":"%s","load_end":"%s"}\n' \
      $S $rc $START $(dt $T0 $T1) $same_kernel $same_cmd $SRC "$L0" "$L1" >> $LOG
  )
}

allo_one() {  # S
  local S=$1 D=$OUT/allo_S$1 REF=/scratch/hc676/e1_allo_w1/S$1_tile
  rm -rf $D; mkdir -p $D
  (
    export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
    source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
    export PYTHONPATH=/scratch/hc676/e22_autosa_tapa/allo
    cd $D
    local L0=$(load) START=$(date +%FT%T)
    export FE_T0=$(now)
    python3 $W/fe_allo.py $S $D/prj $W > $D/fe.log 2>&1
    local rc=$? T1=$(now) L1=$(load)
    local same=false
    cmp -s $D/prj/kernel.cpp $REF/kernel_allo.cpp && same=true
    local phases=$(grep -h '^FE_ALLO ' $D/fe.log | cut -d' ' -f2-)
    [ -n "$phases" ] || phases='{}'
    printf '{"flow":"allo","size":%d,"rc":%d,"start":"%s","wall_s":%s,"phases":%s,"same_kernel_as_e1":%s,"e1_dir":"%s","load_start":"%s","load_end":"%s"}\n' \
      $S $rc $START $(dt $FE_T0 $T1) "$phases" $same $REF "$L0" "$L1" >> $LOG
  )
}

echo "fe_run start $(date +%FT%T) load $(load)"
autosa_one 4 /scratch/hc676/e1_autosa_matched/S4 ""
for s in 8 16 32; do autosa_one $s /scratch/hc676/e1_autosa_wide/S$s --host-serialize; done
for s in 4 8 16 32; do allo_one $s; done
echo "fe_run done $(date +%FT%T)"
