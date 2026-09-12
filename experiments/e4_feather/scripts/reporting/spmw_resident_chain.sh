#!/bin/bash
# The resident (feather_stream_x) whole-workload runs, each after the streamed
# chain that owns the directory has finished (its done-file), reusing the
# HLS'd roles and the assembled array when a first attempt left them behind.
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build PYTHONDONTWRITEBYTECODE=1
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
cd /scratch/hc676/e4_work
S=/scratch/hc676/e4_spmw
reuse() { [ -f $1/roles.json ] && echo "--reuse-build"; }
until [ -f $S/wl_chainA.log.done ]; do sleep 60; done
python3 e4_spmw_run.py --N 8 --workload gemm --gemm 128,128,128 --pattern small --seed 0 --resident $(reuse $S/wl_gemm_N8_x) --out $S/wl_gemm_N8_x
python3 e4_spmw_run.py --N 16 --workload gemm --gemm 128,128,128 --pattern mid --seed 0 --resident $(reuse $S/wl_gemm_N16_x) --out $S/wl_gemm_N16_x
until [ -f $S/wl_chainB.log.done ]; do sleep 60; done
python3 e4_spmw_run.py --N 4 --workload conv --conv 64,16,16,64 --pattern small --seed 0 --resident $(reuse $S/wl_conv_N4_x) --out $S/wl_conv_N4_x
