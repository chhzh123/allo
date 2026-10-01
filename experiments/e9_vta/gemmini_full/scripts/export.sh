#!/bin/bash
# export.sh: the whole-Gemmini and VTA-Core runs, in the repo's layout.
# Generated SystemVerilog is not exported: the pins and the harness rebuild it.
set -u
G=/scratch/hc676/gemmini_full
V=/scratch/hc676/vta_build/core_bench
X=$G/export/experiments/e9_vta
rm -rf $G/export; mkdir -p $X/gemmini_full/source/harness/src/main/scala/gen $X/gemmini_full/source/project $X/gemmini_full/scripts $X/vta_core
cp $G/build.sbt $X/gemmini_full/source/
cp $G/project/build.properties $X/gemmini_full/source/project/
cp $G/harness/src/main/scala/gen/ElaborateGemmini.scala $X/gemmini_full/source/harness/src/main/scala/gen/
cp $G/elab.sh $G/pnr_gemmini.sh $G/closure.py $G/ramfix.py $G/split.py $G/collect_routes.py $G/run_rocc_xsim.sh \
   $G/run_all_sims.sh $G/run_b128.sh $G/run_phased.sh $G/scope_timing.sh $G/gemmini_scope_timing.tcl $G/after_routes.sh $G/export.sh \
   $X/gemmini_full/scripts/
cp $G/bin/dtc $X/gemmini_full/scripts/dtc
{
  echo "gemmini           $(cd /scratch/hc676/gemmini_src && git rev-parse HEAD)  github.com/ucb-bar/gemmini"
  echo "chipyard (pin)    $(cat /scratch/hc676/gemmini_src/CHIPYARD.hash)  github.com/ucb-bar/chipyard"
  for r in rocket-chip diplomacy cde hardfloat firesim; do
    echo "$(printf '%-17s' $r) $(cd $G/$r && git rev-parse HEAD)  $(cd $G/$r && git remote get-url origin | sed 's|https://||;s|\.git$||')"
  done
  echo "gemmini-rocc-tests $(cd $G/rocc_tests && git rev-parse HEAD)  github.com/ucb-bar/gemmini-rocc-tests (include/gemmini.h only)"
  echo "chisel 6.5.0, scala 2.13.12, firtool 1.62.0"
} > $X/gemmini_full/pins.txt
for d in $G/pnr_*/; do
  n=$(basename $d); n=${n#pnr_}
  grep -q "rc=0" $d/DONE 2>/dev/null || continue
  o=$X/gemmini_full/route/$n; mkdir -p $o
  cp $d/pnr.tcl $d/DONE $d/util.rpt $d/util_hier.rpt $d/files.txt $d/ramfix.txt $o/
  cp $d/scope.txt $d/scope_timing.rpt $o/ 2>/dev/null
  # The summary and the worst path of each group, not every path.
  awk '/^Slack/ { n++ } n <= 1' $d/timing.rpt | head -400 > $o/timing.rpt
  python3 $G/split.py $d > $o/split.txt 2>&1
  v=$(echo $n | cut -d_ -f1); s=$(echo $n | cut -d_ -f2)
  cp $G/out/${v}_$s/gemmini_params.h $o/ 2>/dev/null
done
mkdir -p $X/gemmini_full/sim
for f in $G/sim/*.log; do
  n=$(basename $f .log)
  grep -h "GEMROCC" $f | sed "s/^/$n /" >> $X/gemmini_full/sim/results.txt
  mkdir -p $X/gemmini_full/sim/$n
  cp $G/sim/$n/tb.sv $G/sim/$n/cmd_funct.hex $G/sim/$n/cmd_rs1.hex $G/sim/$n/cmd_rs2.hex $X/gemmini_full/sim/$n/ 2>/dev/null
done
for f in $V/*.log; do
  n=$(basename $f .log)
  grep -h "VTACORE" $f | sed "s/^\(VTACORE PLAN\)/$n \1/" >> $X/vta_core/results.txt
  mkdir -p $X/vta_core/$n
  cp $V/$n/tb.sv $V/$n/ins.hex $V/$n/uop.hex $V/$n/xsim.log $X/vta_core/$n/ 2>/dev/null
done
cp $V/run_core.sh $V/run_all.sh /scratch/hc676/vta_build/scope_relaxed.sh $X/vta_core/
# The whole engine at the two relaxed clock targets; the 3.333 ns route is in core/.
for d in /scratch/hc676/vta_build/pnr_Core_w*_p4?; do
  grep -q "rc=0" $d/DONE 2>/dev/null || continue
  o=$X/vta_core/route/$(basename $d | sed "s/pnr_Core_//"); mkdir -p $o
  cp $d/pnr.tcl $d/DONE $d/util.rpt $d/scope_timing.rpt $o/ 2>/dev/null
  grep -E "^(SCOPE|DATAPATH)_" $d/scope.log > $o/scope.txt 2>/dev/null
  awk "/^Slack/ { n++ } n <= 1" $d/timing.rpt | head -400 > $o/timing.rpt
done
python3 $G/collect_routes.py > $X/gemmini_full/routes.json
du -sh $X/gemmini_full $X/vta_core
