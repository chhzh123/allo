#!/bin/bash
# The SPMW memory-fed kernel at the same port width AutoSA uses: S int8 per
# beat for each operand, so the two differ in the array rather than in how
# much memory bandwidth each was given.
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
cd /scratch/hc676/allo || exit 2
for S in 4 8; do
  W=$((8 * S))
  K=/scratch/hc676/e1_spmw_matched_$S; rm -rf $K ${K}_sim ${K}_ops
  echo "=== S=$S, widen=${W}b (AutoSA feeds $S int8 a beat)"; date +%T
  timeout 5400 python3 -u scripts/spmw_package_kernel.py --design autosa --size $S \
      --frequency 300 --jobs 8 --sim --widen $W --out $K > $K.package.log 2>&1
  echo "  package rc=$?"; grep -E "AXI widths|AXI data widths" $K.package.log | head -1
  timeout 1200 python3 -u scripts/spmw_array_operands.py --design autosa --size $S --out ${K}_ops > $K.ops.log 2>&1
  timeout 5400 python3 -u scripts/spmw_kernel_sim.py $K ${K}_ops --design autosa --size $S > $K.sim.log 2>&1
  echo "  sim rc=$?"; grep -E "SPMW TB" $K.sim.log | head -3
done
echo SPMW_MATCHED_DONE
