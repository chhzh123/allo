#!/bin/bash
# spmw_pnr_chain.sh <design> <wait-file> <sizes...>: the SPMW port through
# spmw_build_array.py --pnr at 300 MHz, one size after another, starting once
# <wait-file> exists (a finished chain's marker) so at most two Vivado jobs of
# this package run at once.
DESIGN=$1; WAITF=$2; shift 2
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build PYTHONDONTWRITEBYTECODE=1
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
until [ -f "$WAITF" ]; do sleep 120; done
cd /scratch/hc676/allo
for N in "$@"; do
  OUT=/scratch/hc676/e4_pnr/spmw_${DESIGN}_${N}
  if [ -f $OUT/E4_DONE ]; then continue; fi
  mkdir -p $OUT
  T0=$(date +%s)
  python3 scripts/spmw_build_array.py --design $DESIGN --size $N --frequency 300 --jobs 8 --pnr --out $OUT > $OUT/build.log 2>&1
  echo "rc=$? seconds=$(( $(date +%s) - T0 ))" > $OUT/E4_DONE
done
echo CHAIN_DONE > /scratch/hc676/e4_pnr/spmw_${DESIGN}_chain.done
