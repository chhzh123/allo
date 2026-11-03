#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# collect.sh: stage the exportable files of the AutoSA-on-TAPA run under stage/autosa_tapa.
set -u
W=/scratch/hc676/e22_autosa_tapa; O=$W/stage/autosa_tapa
rm -rf $W/stage; mkdir -p $O/scripts $O/logs
cp $W/gen.sh $W/compile.sh $W/cosim.sh $W/fastcosim_beats.sh $W/pnr.sh $W/all.sh \
   $W/phases.py $W/util.py $W/hlsc_beats.sh $O/scripts/
cp $W/driver.log $W/compile_times.txt $W/cosim_results.txt $W/pnr_results.txt $O/logs/
grep -h -o "BEATS.*" $W/cosim/S16_probe/cosim.log > $O/logs/S16_write_probe.txt
for S in 4 8 16 32; do
  D=$O/S$S; mkdir -p $D/source $D/generated $D/report
  G=$W/gen/S$S
  cp $G/kernel.c $G/kernel.h $G/out/src/cmd $D/source/
  cp $G/out/src/kernel_kernel.cpp $G/out/src/kernel_kernel.h $G/out/src/kernel_host.cpp \
     $G/out/src/kernel_host.h $D/generated/
  cp $W/build/S${S}_j8/compile.log $D/report/compile_j8.log
  cp $W/build/S${S}_j1/compile.log $D/report/compile_j1.log
  cp $W/build/S${S}_j8/work/report/*_csynth.rpt $D/report/
  grep -v "^\s*$" $W/cosim/S${S}_j8/cosim.log | grep -v "^INFO: \[" > $D/report/cosim.log
  cp $W/pnr/S${S}_j8/util.rpt $W/pnr/S${S}_j8/timing.rpt $W/pnr/S${S}_j8/pnr.tcl $D/report/
done
python3 $W/phases.py $W/build/S*_j8 $W/build/S*_j1 > $O/logs/phases.txt
python3 $W/util.py $W/pnr/S4_j8 $W/pnr/S8_j8 $W/pnr/S16_j8 $W/pnr/S32_j8 > $O/logs/util.txt
du -sh $O; find $O -type f | wc -l
