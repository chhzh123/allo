#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# sycl_compile.sh <reps> <S>...: time Intel's SYCL-to-RTL compile of E1's array
# (src/e1_sycl.cpp) at each S, <reps> times (or repetitions <a>-<b>), the E5 way: a fresh directory a
# compile, one compile at a time, the order of the sizes shuffled in each
# repetition from a fixed seed, and the load average taken before and after.
# The compile is oneAPI 2025.0's early image -- HLS to RTL and its reports,
# no Quartus -- for Agilex 7 at 300 MHz. One row of runs/results.csv a compile.
main() {
# oneAPI's vars.sh reads variables it has not set and resets the positional
# parameters of whatever sources it: keep the arguments, source it, then set -u.
local args=("$@")
source /scratch/hc676/intel/env.sh
set -u
set -- "${args[@]}"
E=/scratch/hc676/e17_sycl
REPS=$1; shift
FIRST=1; LAST=$REPS
case $REPS in *-*) FIRST=${REPS%-*}; LAST=${REPS#*-};; esac
mkdir -p $E/runs
CSV=$E/runs/results.csv
[ -f $CSV ] || echo "rep,side,rc,wall_s,user_s,sys_s,max_rss_kb,load_start,load_end,rtl_files,aoc_start_s,hls_done_s,kernels,cells,frac_dsps,aluts,ffs,started" > $CSV
for rep in $(seq $FIRST $LAST); do
  order=$(python3 -c "import random, sys; r = random.Random($rep); s = sys.argv[1:]; r.shuffle(s); print(' '.join(s))" "$@")
  for s in $order; do
    D=$E/runs/S${s}_r$rep
    rm -rf $D; mkdir -p $D; cd $D || exit 1
    l0=$(cut -d' ' -f1 /proc/loadavg); started=$(date -Iseconds); t0=$(date +%s.%N)
    # -Xsv prints each stage as it ends; stamp.py puts the time in front.
    /usr/bin/time -v -o time.txt icpx -fsycl -fintelfpga -DSIDE=$s -Xshardware -fsycl-link=early \
      -Xstarget=Agilex7 -Xsclock=300MHz -Xsv $E/src/e1_sycl.cpp -o e1.a 2>&1 \
      | python3 -u $E/stamp.py > build.log
    rc=${PIPESTATUS[0]}
    t1=$(date +%s.%N); l1=$(cut -d' ' -f1 /proc/loadavg)
    wall=$(python3 -c "print('%.1f' % ($t1 - $t0))")
    user=$(sed -n 's/.*User time (seconds): //p' time.txt)
    sys=$(sed -n 's/.*System time (seconds): //p' time.txt)
    rss=$(sed -n 's/.*Maximum resident set size (kbytes): //p' time.txt)
    rtl=$(find e1.prj -name "*.sv" -o -name "*.v" 2>/dev/null | wc -l)
    echo "$rep,$s,$rc,$wall,$user,$sys,$rss,$l0,$l1,$rtl,$(python3 $E/sycl_row.py $D),$started" >> $CSV
  done
done
echo "SYCL_COMPILE_DONE $(date -Iseconds)" >> $E/runs/done.txt
}
main "$@"
