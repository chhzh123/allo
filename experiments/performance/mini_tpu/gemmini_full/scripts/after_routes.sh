#!/bin/bash
# after_routes.sh: as each 3.333 ns route finishes, time its execute scope;
# the matmul-only engine is then also routed at a relaxed 6.0 ns target, as
# at 4x4, and that route timed the same way.
cd /scratch/hc676/gemmini_full
one() {
  until [ -f pnr_$1_$2/DONE ]; do sleep 30; done
  ./scope_timing.sh pnr_$1_$2
  if [ "$1" = matmul ]; then
    ./pnr_gemmini.sh matmul $2 6.0 p6
    ./scope_timing.sh pnr_matmul_$2_p6
  fi
}
for v in matmul shift lean; do for d in 8 16; do one $v $d & done; done
./scope_timing.sh pnr_matmul_4_p6 &
wait
echo AFTER_ROUTES_DONE
