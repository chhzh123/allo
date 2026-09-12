#!/bin/bash
# e4b_pnr_worker.sh <first|second>: the SPMW port's P&R (spmw_build_array.py --pnr
# at 300 MHz) in priority order -- feather-stream 4, feather 4, feather-stream 8,
# feather 8, feather-stream 16, feather 16 -- one job at a time per worker. The
# first worker starts when either FEATHER RTL 32x32 P&R chain has finished (its
# Vivado slot is free), the second when both have, so this package never runs
# more than two Vivado jobs at once. A job is claimed with mkdir (atomic), so the
# two workers never take the same one; a claimed job's E4_DONE marker (written
# beforehand by hand, "claimed ...") makes the previous agent's queued chains skip
# it, and is overwritten with the real result when the job ends.
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build PYTHONDONTWRITEBYTECODE=1
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
P=/scratch/hc676/e4_pnr
if [ "$1" = first ]; then
  until [ -f $P/orig_chain.done ] || [ -f $P/fixed_chain.done ]; do sleep 60; done
else
  until [ -f $P/orig_chain.done ] && [ -f $P/fixed_chain.done ]; do sleep 60; done
fi
echo "worker $1 starts $(date)"
cd /scratch/hc676/allo
for job in feather-stream:4 feather:4 feather-stream:8 feather:8 feather-stream:16 feather:16; do
  D=${job%%:*}; N=${job##*:}
  OUT=$P/spmw_${D}_${N}
  mkdir -p $OUT
  mkdir $OUT/lock 2>/dev/null || continue
  echo "== $job $(date)"
  T0=$(date +%s)
  python3 -u scripts/spmw_build_array.py --design $D --size $N --frequency 300 --jobs 8 --pnr --out $OUT > $OUT/build.log 2>&1
  RC=$?
  echo "rc=$RC seconds=$(( $(date +%s) - T0 ))" > $OUT/E4_DONE
  echo "$job rc=$RC $(( $(date +%s) - T0 ))s: $(grep -E "array clock|IMPLEMENTATION OK|Error|ERROR" $OUT/build.log | tail -n 2 | tr '\n' ' ')"
done
echo "worker $1 done $(date)"
