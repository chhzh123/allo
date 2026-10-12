#!/bin/bash
# probe.sh <design> <S> [jobs]: one array cosimulation of the tiled design.
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export PYTHONPATH=/scratch/hc676/allo:/scratch/hc676/allo/tests/dataflow/spmw
D=$1; S=$2; J=${3:-4}
E=/scratch/hc676/e16_tiles
O=$E/spmw/probe_${D}_S$S
rm -rf $O
cd /scratch/hc676/allo
T0=$(date +%s)
python3 -u scripts/spmw_build_array.py --design $D --size $S --frequency 300 --jobs $J --cosim --out $O > $O.log 2>&1
echo "PROBE $D S=$S rc=$? wall_s=$(( $(date +%s) - T0 ))" >> $O.log
