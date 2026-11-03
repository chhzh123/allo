#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# run_one.sh <flow> <S> <jobs> <rep>: one timed frontend + HLS compile in a fresh
# directory, stopping when HLS is done (no packaging, elaboration or P&R); appends
# one JSON line to runs.ndjson and keeps only the logs.
#   tapa         AutoSA --target=autosa_tapa (E1's options) + gen.sh's kernel fixes,
#                then `tapa analyze ... synth -j <jobs> link`: per-task HLS and TAPA's
#                top-level RTL (`link` instruments the top with every task instance),
#                everything `tapa compile` does but the .xo packing
#   spmw_kernel  spmw_package_kernel.py --design autosa --sim: the memory-fed kernel's
#                frontend and its roles' and feeders' HLS, --jobs <jobs>
#   spmw_mesh    spmw_ablate_compile.py --design gemm8 --mode shared-parallel: E5's
#                frontend + HLS measurement of E1's mesh, --jobs <jobs>; each role's
#                HLS job is csynth + export_design, as SPMW's flow runs it
#   spmw_mesh_rtl  the same with export_design removed (spmw_mesh_rtl.py): to RTL only,
#                which is what the array is assembled from
#   spmw_kernel_rtl  spmw_package_kernel.py --sim with export_design removed from
#                every role and feeder (spmw_kernel_rtl.py): the kernel to RTL only
main() {
set -u
local FLOW=$1 S=$2 J=$3 REP=$4
local W=/scratch/hc676/e22_autosa_tapa C=/scratch/hc676/e22_autosa_tapa/cmp
local T=$W/allo D=$C/runs/${FLOW}_S${S}_j${J}_r${REP}
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
# TAPA runs in its own environment (Vitis + system): with the SPMW conda environment
# first on PATH, `tapa analyze` takes 34 s instead of 4 s at 32x32 on the same source.
local TAPA_PATH=$PATH
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
export PYTHONPATH=$T
rm -rf $D; mkdir -p $D $C/logs
local L=$C/logs/${FLOW}_S${S}_j${J}_r${REP}
local load0=$(cut -d" " -f1-3 /proc/loadavg) t0=$(date +%s.%N) t1="" rc=0
case $FLOW in
tapa)
  local G=$D/autosa EXTRA=--host-serialize
  [ $S -eq 4 ] && EXTRA=""
  mkdir -p $G/out/src $G/out/latency_est $G/out/resource_est $G/out/tuning
  cp $W/gen/S$S/kernel.c $W/gen/S$S/kernel.h $W/gen/S$S/simd_info.json $G/
  (cd /scratch/hc676/autosa/AutoSA && PATH=/scratch/hc676/autosa/llvm9/bin:$PATH \
    LD_LIBRARY_PATH=/scratch/hc676/autosa/llvm9/lib:${LD_LIBRARY_PATH:-} \
    ./autosa $G/kernel.c --config=./autosa_config/autosa_config.json --target=autosa_tapa \
    --output-dir=$G/out $EXTRA \
    --sa-sizes="{kernel[]->space_time[3];kernel[]->array_part[$S,$S,$S];kernel[]->latency[1,1];kernel[]->simd[1]}" \
    --simd-info=$G/simd_info.json --hls > $L.autosa.log 2>&1) || rc=1
  sed -i -E 's/tapa::(read_only|write_only)_mmap</tapa::mmap</g' $G/out/src/kernel_kernel.cpp $G/out/src/kernel_kernel.h
  python3 - $G/out/src/kernel_kernel.h <<'PY'
import re, sys
h = open(sys.argv[1]).read()
open(sys.argv[1], "w").write(re.sub(r"^void kernel0\(.*?\);\n", "", h, count=1, flags=re.M | re.S))
PY
  t1=$(date +%s.%N)
  PATH=$TAPA_PATH tapa -w $D/work analyze -f $G/out/src/kernel_kernel.cpp -t kernel0 \
    synth --part-num xcu280-fsvh2892-2L-e --clock-period 3.333 -j $J --keep-hls-work-dir \
    link > $L.log 2>&1 || rc=1
  grep -q "writing generated auxiliary RTL files" $L.log || rc=1;;
spmw_kernel)
  (cd $T && python3 -u scripts/spmw_package_kernel.py --design autosa --size $S --frequency 300 \
    --jobs $J --sim --out $D > $L.log 2>&1) || rc=1
  grep -q "stopping before v++" $L.log || rc=1;;
spmw_mesh)
  rmdir $D
  (cd $T && python3 -u scripts/spmw_ablate_compile.py --design gemm8 --size $S \
    --mode shared-parallel --jobs $J --out $D > $L.log 2>&1) || rc=1
  [ -f $D/e5.json ] && cp $D/e5.json $L.e5.json || rc=1;;
spmw_kernel_rtl)
  (cd $T && python3 -u $C/spmw_kernel_rtl.py $T --size $S --jobs $J --out $D > $L.log 2>&1) || rc=1
  grep -q "stopping before v++" $L.log || rc=1;;
spmw_mesh_rtl)
  rmdir $D
  (cd $T && python3 -u $C/spmw_mesh_rtl.py $T --design gemm8 --size $S --jobs $J \
    --out $D > $L.log 2>&1) || rc=1
  grep -q "^E5 " $L.log || rc=1;;
esac
local t2=$(date +%s.%N) load1=$(cut -d" " -f1-3 /proc/loadavg)
python3 $C/parse_run.py "$FLOW" "$S" "$J" "$REP" "$rc" "$t0" "${t1:-$t0}" "$t2" "$load0" "$load1" "$D" "$L" >> $C/runs.ndjson
rm -rf $D
}
main "$@"
