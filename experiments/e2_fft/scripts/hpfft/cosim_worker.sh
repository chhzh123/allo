#!/bin/bash
# Serial worker for the seeded (numpy seeds 0/1/2) cosimulations -- one vitis_hls/xsim at a time.
# HP-FFT: fresh <cfg>_s copies (FFT.cpp, FFT.h, project.tcl, shipped testbench) -> run_hls.sh (csim+csynth) ->
# run_cosim5.sh (NT=33 transforms). Allo: run_allocosim_s.sh (fresh strided_n<N>_wrap_s, NT=6).
E=/scratch/hc676/e2_hpfft; A=/scratch/hc676/e2_allo
touch $E/cosim_worker.active; echo $$ > $E/cosim_worker.pid
for job in "n256 UF1" "n1024 UF1" "n128 UF1" "n512 UF1" "n256 UF2" "n1024 UF2" "n256 UF4" "n1024 UF4"; do
  set -- $job; size=$1; cfg=$2; D=$E/$size/${cfg}_s; B=$E/$size/$cfg
  echo "$(date +%FT%T) start $size/${cfg}_s"
  rm -rf $D; mkdir -p $D
  cp $B/FFT.cpp $B/FFT.h $B/project.tcl $D/
  if [ -f $B/testbench_orig.cpp ]; then cp $B/testbench_orig.cpp $D/testbench.cpp; else cp $B/testbench.cpp $D/testbench.cpp; fi
  bash $E/run_hls.sh $size ${cfg}_s > $E/logs/s_hls_${size}_${cfg}.log 2>&1
  bash $E/run_cosim5.sh $size ${cfg}_s > $E/logs/s_cosim5_${size}_${cfg}.log 2>&1
  echo "$(date +%FT%T) done $size/${cfg}_s: $(tr '\n' ' ' < $D/cosim.done)"
done
for N in 128 256 512 1024; do
  echo "$(date +%FT%T) start allo n$N wrap seeded"
  bash $E/run_allocosim_s.sh $N wrap > $E/logs/s_allocosim_${N}_wrap.log 2>&1
  echo "$(date +%FT%T) done allo n$N wrap: $(cat $A/strided_n${N}_wrap_s/hls.done)"
done
rm -f $E/cosim_worker.active
echo "$(date +%FT%T) COSIM_WORKER_EXIT"
