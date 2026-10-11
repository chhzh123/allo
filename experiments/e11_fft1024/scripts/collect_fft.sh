#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# collect_fft.sh: stage what reproduces the N = 1024 FFT table, in E2's shape:
# <framework>/<config>/{source,generated,report}. Copies; touches nothing it
# reads. HP-FFT's fixed UF4 and UF8 come from the builds of 2026-10-03 in
# /scratch/hc676/hpfft_fix and /scratch/hc676/hpfft_pnr: the same scripts as
# hp_run.sh, and sources that mkvariant.py reproduces byte for byte
# (verify_hpfft.txt records the comparison).
set -u
E=/scratch/hc676/e14_fft
T=/scratch/hc676/allo
STAGE=$E/stage
rm -rf $STAGE
c() { cp "$@" 2>/dev/null; }
# ---- SPMW, the lane design ---------------------------------------------------
for W in 2 4 8 16 32; do
  B=$E/spmw_n1024w$W
  [ -d $B ] || continue
  D=$STAGE/spmw/W$W; mkdir -p $D/source $D/generated/fabric $D/generated/roles $D/report/roles
  c $T/tests/dataflow/spmw/test_spmw_fft_lanes.py $D/source/lanes.py
  echo "python3 scripts/spmw_build_array.py --design fftlanes --size 1024 --lanes $W --frequency 300 --jobs 6 --cosim --pnr --out <dir>" > $D/source/command.txt
  c $B/spmw_top.sv $B/spmw_fifo.sv $B/spmw_harness.sv $B/spmw_const.sv $D/generated/fabric/
  for r in $B/*_r[0-9]*; do
    [ -d $r ] || continue
    n=$(basename $r)
    c $r/kernel.cpp $D/generated/roles/$n.cpp
    c $r/prj/sol/syn/report/csynth.rpt $D/report/roles/${n}_csynth.rpt
  done
  c $B/util.rpt $B/timing.rpt $B/route.rpt $B/util_synth.rpt $B/cost.json $D/report/
  c $E/logs/spmw_n1024w$W.log $D/report/build.log
  grep -hE "SPMW (XFORM|COSIM|CYCLES)" $B/sim/xsim.log > $D/report/xform.txt 2>/dev/null
done
# ---- HP-FFT ------------------------------------------------------------------
hp() {  # <config> <hls dir> <pnr dir> <log>
  local D=$STAGE/hpfft/$1 H=$2 P=$3
  [ -d $H ] || return
  mkdir -p $D/source $D/report
  c $H/FFT.cpp $H/FFT.h $H/project.tcl $D/source/
  c $(dirname $(dirname $H))/common.tcl $D/source/
  c $H/build/FFT_300MHz/syn/report/csynth.rpt $D/report/
  c $H/cosim_events.json $H/validate_rtl_vcd.json $H/e2_stimulus.json $H/cosim.tcl $D/report/
  for f in util.rpt timing.rpt route_status.rpt util_synth.rpt util_hier.rpt impl.tcl clock.xdc; do
    c $P/$f $D/report/pnr_$f
  done
  c $4 $D/report/build.log
}
hp UF2 $E/hpfft/n1024/UF2_base $E/hpfft/pnr/n1024_UF2_base $E/hpfft/logs/n1024_UF2_base.log
hp UF4_a1 $E/hpfft/n1024/UF4_a1 $E/hpfft/pnr/n1024_UF4_a1 $E/hpfft/logs/n1024_UF4_a1.log
hp UF4_a8 /scratch/hc676/hpfft_fix/n1024/UF4_a8 /scratch/hc676/hpfft_pnr/n1024_UF4_a8 /scratch/hc676/hpfft_pnr/n1024_UF4_a8.done
hp UF8_b4 /scratch/hc676/hpfft_fix/n1024/UF8_b4 /scratch/hc676/hpfft_pnr/n1024_UF8_b4 /scratch/hc676/hpfft_pnr/n1024_UF8_b4.done
# The two reused builds, checked against a fresh mkvariant.py of the shipped sources.
{ cd $E/hpfft/verify 2>/dev/null && for v in UF4_a8 UF8_b4; do
    for f in FFT.cpp FFT.h project.tcl; do
      cmp -s $v/$f /scratch/hc676/hpfft_fix/n1024/$v/$f && echo "$v/$f identical to mkvariant.py's" || echo "$v/$f DIFFERS"
    done
  done
  for f in run_hls.sh run_cosim.sh common.tcl cosim_events.py gen_stimulus.py testbench_e2s.cpp mkvariant.py; do
    diff <(sed "s#/scratch/hc676/e14_fft/hpfft#ROOT#g" $E/hpfft/$f) <(sed "s#/scratch/hc676/hpfft_fix#ROOT#g" /scratch/hc676/hpfft_fix/$f) > /dev/null \
      && echo "$f identical but for the build root" || echo "$f DIFFERS"
  done; } > $STAGE/hpfft/verify_hpfft.txt
mkdir -p $STAGE/scripts
c $E/fft_build.sh $E/hp_run.sh $E/collect_fft.sh $STAGE/scripts/
find $STAGE -type f -empty -delete
du -sh $STAGE; find $STAGE -type f | wc -l
