#!/bin/bash
# E3 on the board, with the restarted run handle: GPT-2 medium timed (4 reps) then LLaMA-7B (2 reps), every launch checked.
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
cd /scratch/hc676/allo
echo "=== gpt2-medium timed2"; date +%T
rm -rf /scratch/hc676/e3_block_timed2
timeout 7200 python3 scripts/spmw_gpt_block.py --xclbin /scratch/hc676/gptkern_302/spmw_kernel.xclbin --args /scratch/hc676/gptkern_302/args.json --out /scratch/hc676/e3_block_timed2 --seed 0 --reps 4 2>&1 | grep -vE "^\s*$" | tail -60 | cut -c1-170
echo "rc=${PIPESTATUS[0]}"
echo "=== llama-7b device"; date +%T
rm -rf /scratch/hc676/e3_llama_dev
timeout 14400 python3 scripts/spmw_gpt_block.py --model llama-7b --xclbin /scratch/hc676/gptkern_302/spmw_kernel.xclbin --args /scratch/hc676/gptkern_302/args.json --out /scratch/hc676/e3_llama_dev --seed 0 --reps 2 2>&1 | grep -vE "^\s*$" | tail -40 | cut -c1-170
echo "rc=${PIPESTATUS[0]}"; date +%T
echo E3_BOARD2_DONE
