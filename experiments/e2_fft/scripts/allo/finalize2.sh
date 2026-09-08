#!/bin/bash
# Re-assemble the E2 FFT package from whatever has finished (safe to re-run at any time).
PY=/scratch/hc676/allo-agent/bin/python3; E=/scratch/hc676/e2_hpfft; OUT=/scratch/hc676/spmw_eval_remaining_2026-09-06/e2_fft
for sub in hpfft allo; do   # preserve the previous agent's write-up once
  [ -f $OUT/$sub/README_previous_agent.md ] || cp $OUT/$sub/README.md $OUT/$sub/README_previous_agent.md
  [ -f $OUT/$sub/results_previous_agent.csv ] || cp $OUT/$sub/results.csv $OUT/$sub/results_previous_agent.csv
done
$PY $E/make_results2.py && $PY $E/render_tables2.py || exit 1
for sub in hpfft allo; do
  { cat $E/README_${sub}_final.md; echo; echo "## Results tables (rendered from results.csv / hls_estimates.csv on $(date '+%F %H:%M %Z'); re-run /scratch/hc676/e2_hpfft/finalize2.sh to refresh)"; cat $OUT/$sub/summary.md; } > $OUT/$sub/README.md
done
mkdir -p $OUT/hpfft/scripts $OUT/allo/scripts $OUT/hpfft/reports/logs $OUT/allo/reports/logs
cp $E/{common.tcl,run_hls.sh,run_cosim4.sh,run_cosim5.sh,run_pnr.sh,pnr_ooc.sh,queue_hls.sh,pnr_worker.sh,cosim_worker.sh,gen_size.py,gen_stimulus.py,testbench_e2.cpp,testbench_e2_array.cpp,testbench_e2s.cpp,validate_fft.py,validate_fft2.py,cosim_events.py,harvest_hls.py,make_results.py,make_results2.py,render_tables2.py,finalize2.sh,status.py} $OUT/hpfft/scripts/ 2>/dev/null
cp /scratch/hc676/e2_allo/{gen_projects.py,tb.cpp} $E/{tb_s.cpp,run_allohls2.sh,run_allocosim_s.sh,run_pnrallo.sh,pnr_ooc.sh,gen_stimulus.py,validate_fft2.py,make_results2.py,render_tables2.py,finalize2.sh} $OUT/allo/scripts/ 2>/dev/null
cp $E/logs/{queue_hls.log,queue_pnr.log,pnr_worker.log,cosim_worker.log} $OUT/hpfft/reports/logs/ 2>/dev/null
cp $E/logs/{queue_hls.log,queue_pnr.log,pnr_worker.log,cosim_worker.log} $OUT/allo/reports/logs/ 2>/dev/null
rm -f $OUT/hpfft/README_narrative.md $OUT/allo/README_narrative.md
echo "finalized $(date)"
