#!/bin/bash
# The conv workload with the published loader and with the row-wise one.
# The GEMM comparison was re-measured after the loader fix and the conv one
# was not, so its FEATHER column still carries an N^3 weight feed.
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
W=/scratch/hc676/feather_conv
PY=/scratch/hc676/allo-agent/bin/python3
run () {  # run <tag> <rtl> <loader> <N> <mode>
  local tag=$1 rtl=$2 loader=$3 N=$4 mode=$5
  echo "### $tag rtl=$rtl loader=$loader N=$N mode=$mode $(date +%T)"
  $PY -u $W/work/e4_rtl_run.py --rtl $W/$rtl --N $N --out $W/runs/$tag \
      --workload conv --conv 64,16,16,64 --pattern full --seed 0 --mode $mode \
      --zpa 0 --zpw 0 --loader $loader 2>&1 | tail -3
  echo
}
mkdir -p $W/runs
for N in "$@"; do
  run conv_base_N${N}_m1 RTL_base pe  $N 1
  run conv_row_N${N}_m1  RTL_new  row $N 1
done
echo "CONV DONE $(date +%T)"
