#!/bin/bash
# Regenerate results.csv, reports/, validation/, summary.md and README.md for both systems from whatever has
# finished so far (safe to re-run at any time while the queues are still working).
PY=/scratch/hc676/allo-agent/bin/python3; E=/scratch/hc676/e2_hpfft; OUT=/scratch/hc676/spmw_eval_remaining_2026-09-06/e2_fft
$PY $E/make_results.py && $PY $E/render_tables.py
for sub in hpfft allo; do
  { cat $OUT/$sub/README_narrative.md; echo; echo "## Results (rendered from results.csv on $(date +%F\ %H:%M) — re-run /scratch/hc676/e2_hpfft/finalize.sh to refresh)"; cat $OUT/$sub/summary.md; } > $OUT/$sub/README.md
done
cp $E/{common.tcl,run_hls.sh,run_cosim4.sh,run_pnr.sh,pnr_ooc.sh,run_allohls2.sh,run_pnrallo.sh,queue_hls.sh,gen_size.py,testbench_e2.cpp,testbench_e2_array.cpp,validate_fft.py,cosim_events.py,harvest_hls.py,make_results.py,render_tables.py,finalize.sh} $OUT/hpfft/scripts/ 2>/dev/null || { mkdir -p $OUT/hpfft/scripts; cp $E/{common.tcl,run_hls.sh,run_cosim4.sh,run_pnr.sh,pnr_ooc.sh,run_allohls2.sh,run_pnrallo.sh,queue_hls.sh,gen_size.py,testbench_e2.cpp,testbench_e2_array.cpp,validate_fft.py,cosim_events.py,harvest_hls.py,make_results.py,render_tables.py,finalize.sh} $OUT/hpfft/scripts/; }
mkdir -p $OUT/allo/scripts; cp /scratch/hc676/e2_allo/{gen_projects.py,tb.cpp} $OUT/allo/scripts/
echo "finalized $(date)"
