#!/bin/bash
# E4 row-wise loader, part 2: the full single-tile validation matrix over the
# row loader (every BIRRD program, operand pattern, zero point and seed the
# baseline used), and the MODE 0 feed workloads where the load is paid per tile.
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
W=/scratch/hc676/feather_loader
PY=/scratch/hc676/allo-agent/bin/python3
gemm () {  # gemm <tag> <rtl> <loader> <N> <mode>
  echo "### $1  rtl=$2 loader=$3 N=$4 mode=$5  $(date +%T)"
  $PY -u $W/work/e4_rtl_run.py --rtl $W/$2 --N $4 --out $W/runs/$1 \
      --workload gemm --gemm 128,128,128 --pattern full --seed 0 --mode $5 \
      --zpa 0 --zpw 0 --loader $3 2>&1 | tail -2
  echo
}
mkdir -p $W/runs $W/matrix
# MODE 0: a weight feed per tile, so the load is paid 32768 / 4096 / 512 times
for N in 4 8 16; do gemm rowfeed_N$N RTL_new row $N 0; done
gemm basefeed_N4 RTL_base pe 4 0
# the validation matrix, row loader, every size
for N in 4 8 16; do
  echo "### validate N=$N  $(date +%T)"
  $PY -u $W/work/e4_matrix.py --rtl $W/RTL_new --N $N --tag row --set validate \
      --loader row --out $W/matrix 2>&1 | tail -2
  echo
done
echo "ALL DONE $(date +%T)"
