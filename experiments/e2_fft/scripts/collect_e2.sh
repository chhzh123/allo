#!/bin/bash
# Gather the E2 files that reproduce its table, in E1's layout:
#   <framework>/N<size>/{source,generated,report}
# Selected rather than mirrored: Allo's cosim hls.log alone is 5.3 MB.
set -u
E=/scratch/hc676/spmw_eval_remaining_2026-09-06/e2_fft
OUT=/scratch/hc676/e2_organised
rm -rf $OUT
take() { [ -f "$1" ] && install -D -m644 "$1" "$2" 2>/dev/null; }
for N in 128 256 512 1024; do
  # ---- SPMW: reports only; its generated code is staged separately ----
  D=$OUT/spmw/N$N
  for f in build.log cost.json xsim.log; do
    take "$E/spmw/reports/N${N}_cosim/$f" "$D/report/cosim_$f"
  done
  for f in build.log cost.json util.rpt timing.rpt route.rpt util_synth.rpt; do
    take "$E/spmw/reports/N${N}_pnr/$f" "$D/report/pnr_$f"
  done
  # ---- Allo: the emitted HLS is the generated artefact ----
  D=$OUT/allo/N$N
  C=$E/allo/reports/allo-strided_n${N}_wrap_csynth
  take "$C/kernel.cpp" "$D/generated/kernel.cpp"
  take "$C/kernel.h"   "$D/generated/kernel.h"
  for f in run_e2.tcl tb.cpp; do take "$C/$f" "$D/source/$f"; done
  for f in csynth.rpt fft_csynth.rpt hls.done; do take "$C/$f" "$D/report/$f"; done
  S=$E/allo/reports/allo-strided_n${N}_wrap_cosim
  for f in fft_cosim.rpt lat.rpt result.transaction.rpt e2_stimulus.json; do
    take "$S/$f" "$D/report/$f"
  done
  P=$E/allo/reports/allo-strided_n${N}_wrap_pnr
  for f in util.rpt timing.rpt route_status.rpt util_synth.rpt clock.xdc impl.tcl; do
    take "$P/$f" "$D/report/pnr_$f"
  done
  # ---- HP-FFT: hand-written HLS; 128 and 512 are derived from the shipped sizes ----
  D=$OUT/hpfft/N$N
  case $N in 256|1024) PRE=hpfft_n${N}_UF1;; *) PRE=hpfft-derived_n${N}_UF1;; esac
  C=$E/hpfft/reports/${PRE}_csynth
  for f in FFT.cpp FFT.h project.tcl common.tcl gen_size.py; do take "$C/$f" "$D/source/$f"; done
  for f in csynth.rpt harvest.json warnings.json; do take "$C/$f" "$D/report/$f"; done
  S=$E/hpfft/reports/${PRE}_cosim
  for f in FFT_TOP_cosim.rpt cosim.log cosim_events.json e2_stimulus.json e2_xsim.log testbench.cpp cosim.tcl; do
    take "$S/$f" "$D/report/$f"
  done
  P=$E/hpfft/reports/${PRE}_pnr
  for f in util.rpt timing.rpt route_status.rpt util_synth.rpt clock.xdc impl.tcl; do
    take "$P/$f" "$D/report/pnr_$f"
  done
done
echo "=== collected ==="; du -sh $OUT; find $OUT -type f | wc -l
for fw in spmw allo hpfft; do
  printf "  %-7s " $fw
  for N in 128 256 512 1024; do
    printf "N%s:%s " $N "$(find $OUT/$fw/N$N -type f 2>/dev/null | wc -l)"
  done; echo
done
