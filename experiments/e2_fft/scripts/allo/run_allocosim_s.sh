#!/bin/bash
# usage: run_allocosim_s.sh <N> <wrap|raw>  -- seeded csim + csynth + cosim of the Allo strided FFT in a fresh copy
# strided_n<N>_<var>_s of the previous project dir (Allo's kernel.cpp/kernel.h/run.tcl as generated, kernel.cpp with
# the depth= edit on the m_axi pragmas); tb.cpp = tb_s.cpp reads NT transforms of numpy stimulus (gen_stimulus.py,
# seeds 0/1/2). The previous project (LCG stimulus, NT=4) is left untouched. Same tcl as run_allohls2.sh otherwise.
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
ulimit -n 8192
A=/scratch/hc676/e2_allo; E=/scratch/hc676/e2_hpfft; PY=/scratch/hc676/allo-agent/bin/python3
SRC=$A/strided_n$1_$2; P=${SRC}_s; NT=${NT:-6}
rm -rf $P; mkdir -p $P
for f in kernel.cpp kernel.h run.tcl host.cpp hls.cfg; do cp $SRC/$f $P/; done
cd $P || exit 1
$PY $E/gen_stimulus.py $1 $NT $P/e2_stimulus || exit 1
sed -e "s|#define E2_STIM \"e2_stimulus.txt\"|#define E2_STIM \"$P/e2_stimulus.txt\"|" $E/tb_s.cpp > tb.cpp
sed -e "s|add_files -tb host.cpp -cflags \"-std=gnu++0x\"|add_files -tb tb.cpp -cflags \"-std=gnu++0x -DE2_N=$1 -DE2_NT=$NT\"|" \
    -e "s|^csynth_design|csim_design\ncsynth_design\ncosim_design -trace_level port -rtl verilog|" run.tcl > run_e2s.tcl
T0=$(date +%s)
timeout -k 120 21600 vitis_hls -f run_e2s.tcl > hls.log 2>&1
RC=$?
echo "rc=$RC hls_wall_s=$(( $(date +%s) - T0 )) start=$T0 end=$(date +%s) NT=$NT stimulus=numpy_seeds_0_1_2" > hls.done
S=$P/out.prj/solution1
for w in wrapc wrapc_pc; do
  [ -f $S/sim/$w/e2_outputs.txt ] && $PY $E/validate_fft2.py $S/sim/$w/e2_inputs.txt $S/sim/$w/e2_outputs.txt $1 $NT bitrev --seeds $P/e2_stimulus.json > $P/validate_$w.json 2>&1
done
exit 0
