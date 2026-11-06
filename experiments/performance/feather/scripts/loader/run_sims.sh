#!/bin/bash
# E4 row-wise loader: baseline vs new at 4x4, 8x8, 16x16, plus the crossed
# pairings that must fail (image and hardware disagreeing).
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
W=/scratch/hc676/feather_loader
PY=/scratch/hc676/allo-agent/bin/python3
run () {  # run <tag> <rtl> <loader> <N>
  local tag=$1 rtl=$2 loader=$3 N=$4
  echo "### $tag  rtl=$rtl loader=$loader N=$N  $(date +%T)"
  $PY -u $W/work/e4_rtl_run.py --rtl $W/$rtl --N $N --out $W/runs/$tag \
      --workload gemm --gemm 128,128,128 --pattern full --seed 0 --mode 1 \
      --zpa 0 --zpw 0 --loader $loader 2>&1 | tail -3
  echo
}
mkdir -p $W/runs
for N in 4 8 16; do run base_N$N RTL_base pe $N; done
for N in 4 8 16; do run row_N$N  RTL_new  row $N; done
# the crossed pairs: each must fail, or the check proves nothing
run cross_oldrtl_newimg_N4 RTL_base row 4
run cross_newrtl_oldimg_N4 RTL_new  pe  4
echo "ALL DONE $(date +%T)"
