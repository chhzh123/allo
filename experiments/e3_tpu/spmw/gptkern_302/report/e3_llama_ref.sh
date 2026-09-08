#!/bin/bash
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
cd /scratch/hc676/allo; rm -rf /scratch/hc676/e3_llama_ref
timeout 7200 python3 scripts/spmw_gpt_block.py --model llama-7b --no-device --out /scratch/hc676/e3_llama_ref --seed 0 2>&1 | grep -vE "^\s*$" | tail -24 | cut -c1-170
echo "rc=${PIPESTATUS[0]}"; echo E3_LLAMA_REF_DONE
