#!/bin/bash
# usage: run_pnrallo.sh <N> <wrap|raw> -- OOC P&R of the Allo strided FFT RTL (waits for its HLS run)
A=/scratch/hc676/e2_allo; P=$A/strided_n$1_$2
for i in $(seq 1 720); do [ -f $P/hls.done ] && break; sleep 20; done
bash /scratch/hc676/e2_hpfft/pnr_ooc.sh $P/out.prj/solution1/syn/verilog fft $P/pnr $P/pnr.done
