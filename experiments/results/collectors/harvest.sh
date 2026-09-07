#!/bin/bash
# Re-run every package's own collector, so results that land after its agent
# stopped still reach the bundle. Each collector is the one that package's
# agent wrote and used; failures are reported, not hidden.
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
cd /scratch/hc676/allo
B=/scratch/hc676/spmw_eval_remaining_2026-09-06
run() { echo "=== $1"; shift; timeout 900 "$@" > /tmp/harvest_$$.log 2>&1; echo "  rc=$? $(tail -2 /tmp/harvest_$$.log | tr '\n' ' ' | cut -c1-200)"; }
[ -f /scratch/hc676/e1_autosa_2026-09-06/scripts/collect.py ] && run autosa python3 /scratch/hc676/e1_autosa_2026-09-06/scripts/collect.py
[ -f /scratch/hc676/e4b_work/e4b_collect.py ] && run feather python3 /scratch/hc676/e4b_work/e4b_collect.py
[ -f /scratch/hc676/e6_work/e6_collect2.py ] && run attention python3 /scratch/hc676/e6_work/e6_collect2.py
[ -f /scratch/hc676/e2_hpfft/make_results2.py ] && run hpfft python3 /scratch/hc676/e2_hpfft/make_results2.py
[ -f /scratch/hc676/e2_allo/make_results2.py ] && run allofft python3 /scratch/hc676/e2_allo/make_results2.py
# The Allo package is owned by a live agent; run whichever collector it is
# currently maintaining (the newest), never an older copy, or a harvest
# overwrites its results with a stale layout.
allo_collect=$(ls -t /scratch/hc676/e1_allo_w1/drv/e1_allo_collect.py /scratch/hc676/e1_allo_drv3/e1_allo_collect.py /scratch/hc676/e1_allo_drv2/e1_allo_collect.py 2>/dev/null | head -1)
[ -n "$allo_collect" ] && run allo_gemm python3 "$allo_collect"
[ -f /scratch/hc676/e5/e5_runs.log ] && run compile python3 /scratch/hc676/e5_rows.py /scratch/hc676/e5/e5_runs.log /scratch/hc676/spmw_eval_remaining_2026-09-06/e5_compile
run bundle python3 /scratch/hc676/bundle_assemble.py /scratch/hc676/spmw_eval_remaining_2026-09-06
echo "=== row counts:"
for p in e1_gemm/spmw e1_gemm/spmw_mem e1_gemm/autosa e1_gemm/allo e2_fft/spmw e2_fft/hpfft e2_fft/allo e3_tpu e4_feather e6_attention; do
  f=$B/$p/results.csv
  [ -f $f ] && echo "  $p: $(( $(wc -l < $f) - 1 )) rows, pass=$(grep -coE ',(pass|ok),' $f)" || echo "  $p: no results.csv"
done
echo HARVEST_DONE
