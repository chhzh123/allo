#!/bin/bash
# E5: compilation time, three modes x three repetitions, mode order randomised
# per (repetition, design, size) from a fixed seed, fresh directory per run,
# eight workers. Waits for the E1/E2 sweeps (the frozen designs) to finish
# first so their P&R runs are not disturbed and the machine is as quiet as it
# gets. One line per run in e5_runs.log ("E5 key=value ...").
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
cd /scratch/hc676/allo
W=/scratch/hc676/e5; mkdir -p $W
B=/scratch/hc676/spmw_eval_remaining_2026-09-06/e5_compile; mkdir -p $B/reports
for m in E1_SPMW_DONE:/scratch/hc676/e1_spmw.log E1_SPMWMEM_DONE:/scratch/hc676/e1_spmw_mem.log E2_SDF_DONE:/scratch/hc676/e2_sdf.log; do
  marker=${m%%:*}; log=${m#*:}
  until grep -q $marker $log 2>/dev/null; do sleep 120; done
done
echo "sweeps finished, starting E5 at $(date +%T), load $(cut -d' ' -f1 /proc/loadavg)"
POINTS="gemm8:4 gemm8:8 gemm8:16 gemm8:32 fftsdf:128 fftsdf:256 fftsdf:512 fftsdf:1024"
for rep in 0 1 2; do
  for point in $POINTS; do
    d=${point%%:*}; s=${point#*:}
    # a seeded permutation of the three modes
    order=$(python3 -c "
import random; r = random.Random(hash(('e5', $rep, '$d', $s)) & 0xffffffff)
m = ['shared-serial', 'shared-parallel', 'per-instance']; r.shuffle(m); print(' '.join(m))")
    for mode in $order; do
      run=${d}_${s}_${mode}_r$rep; D=$W/$run; rm -rf $D
      # the per-instance mode at the largest GEMM is bounded by a preset hour
      t=7200; [ "$mode" = per-instance ] && [ "$d" = gemm8 ] && [ "$s" = 32 ] && t=3600
      echo "=== $run (timeout ${t}s) $(date +%T) load $(cut -d' ' -f1 /proc/loadavg)"
      T0=$(date +%s)
      timeout $t python3 -u scripts/spmw_ablate_compile.py --design $d --size $s --mode $mode --jobs 8 --out $D > $D.log 2>&1; rc=$?
      echo "rc=$rc wall=$(( $(date +%s) - T0 ))s completed_jobs=$(grep -cE '^  \S+: rc=0' $D.log)"
      grep -E "^E5 " $D.log >> $W/e5_runs.log
      [ $rc -ne 0 ] && echo "E5 run=$run rc=$rc timeout=$t wall=$(( $(date +%s) - T0 )) completed_jobs=$(grep -cE '^  \S+: rc=0' $D.log) status=timeout_or_fail" >> $W/e5_runs.log
      mkdir -p $B/reports/$run; cp $D.log $B/reports/$run/run.log; cp $D/e5.json $B/reports/$run/ 2>/dev/null
      # the projects are large; keep one role's HLS report as evidence and drop the rest
      find $D -maxdepth 1 -mindepth 1 -type d | head -1 | while read r; do cp -r $r/prj/sol/syn/report $B/reports/$run/$(basename $r)_syn_report 2>/dev/null; done
      rm -rf $D
    done
  done
done
cp $W/e5_runs.log $B/
echo E5_DONE
