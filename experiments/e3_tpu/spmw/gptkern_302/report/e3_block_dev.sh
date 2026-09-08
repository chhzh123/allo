#!/bin/bash
# E3: the GPT-2 medium block chained on the 302 bitstream, one repetition, every launch checked.
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
cd /scratch/hc676/allo
while pgrep -u hc676 -f "^/usr/bin/python3 -u scripts/spmw_board_gpt" >/dev/null; do sleep 30; done
rm -rf /scratch/hc676/e3_block_dev1
timeout 7200 python3 scripts/spmw_gpt_block.py --xclbin /scratch/hc676/gptkern_302/spmw_kernel.xclbin --args /scratch/hc676/gptkern_302/args.json --out /scratch/hc676/e3_block_dev1 --seed 0 --reps 1 2>&1 | tail -22 | cut -c1-170
echo "rc=${PIPESTATUS[0]}"
echo E3_BLOCK_DEV_DONE
