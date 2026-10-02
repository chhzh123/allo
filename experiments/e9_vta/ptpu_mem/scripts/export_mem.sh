#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# export_mem.sh: the whole engine's builds, routes and memory-bench runs, in
# the repo's layout. The microbenchmark build is the routed one; the five-GEMM
# build is the same hardware, kept only where its generated code differs; a
# `p30_` report is the same HLS output routed again at a 3.0 ns target.
set -u
E=/scratch/hc676/e13_mem
X=$E/export/experiments/e9_vta
SRC=/scratch/hc676/allo/tests/dataflow/spmw/test_spmw_ptpu_mem.py
rm -rf $E/export; mkdir -p $X
source /scratch/hc676/e9_lean/export_fns.sh
cd $E || exit 2
bash $E/routes.sh > $E/routes.txt
for s in 4 8 16; do
  engine ptpu_mem $s b_ptpumem-micro_$s mixed_ b_ptpumem-mixed_$s
  R=$X/ptpu_mem/S$s/report
  bash $E/same_hw.sh $s > $R/same_hardware.txt
  for f in area_by_unit.txt util_hier.rpt worst_paths.rpt; do
    [ -f $E/b_ptpumem-micro_$s/$f ] && cp $E/b_ptpumem-micro_$s/$f $R/
  done
  for d in $(ls -d $E/route_${s}_p* 2>/dev/null); do
    p=$(basename $d); p=${p#route_${s}_}
    for f in util.rpt util_synth.rpt util_hier.rpt timing.rpt route.rpt area_by_unit.txt worst_paths.rpt clock.xdc assemble.tcl; do
      [ -f $d/$f ] && cp $d/$f $R/${p}_$f
    done
    grep -E "SPMW (STAGE|UNROUTED)|ARRAY WNS" $d/vivado.log | grep -v "puts" > $R/${p}_pnr_stages.txt
  done
done
mkdir -p $X/ptpu_mem/scripts
for d in $(ls -d $E/bench/*/ | sort); do grep -h SPMWMEM $d/bench.log; done > $X/ptpu_mem/results.txt
cp $E/routes.txt $X/ptpu_mem/routes.txt
cp $E/build.sh $X/ptpu_mem/scripts/spmw_build.sh
cp $E/bench.sh $E/same_hw.sh $E/hier_mem.sh $E/reroute.sh $E/routes.sh $E/export_mem.sh $X/ptpu_mem/scripts/
du -sh $X/ptpu_mem; find $X/ptpu_mem -type f | wc -l
echo done
