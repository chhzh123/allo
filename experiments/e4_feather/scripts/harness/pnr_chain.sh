#!/bin/bash
# pnr_chain.sh <variant name> <RTL dir> <sizes...>: sequential OOC P&R runs.
VAR=$1; RTL=$2; shift 2
for N in "$@"; do
  OUT=/scratch/hc676/e4_pnr/${VAR}_${N}
  if [ -f $OUT/result.txt ] && grep -q E4_PNR_DONE $OUT/result.txt; then continue; fi
  /scratch/hc676/e4_work/pnr_ooc.sh $RTL $N $OUT
done
echo CHAIN_DONE > /scratch/hc676/e4_pnr/${VAR}_chain.done
