#!/bin/bash
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
cd /scratch/hc676/allo
python3 scripts/spmw_package_kernel.py --design gptstage_v1 --size 16 --out /scratch/hc676/gptkern_302 --frequency 300 --link-frequency 300 --vivado-prop run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=ExtraTimingOpt --vivado-prop run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore
echo "BUILD_302_DONE rc=$?"
