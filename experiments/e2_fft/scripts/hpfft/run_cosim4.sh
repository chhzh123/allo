#!/bin/bash
# usage: run_cosim4.sh <size> <cfg>
# Stream configs (UFx): `cosim_design -setup -trace_level port` (runs the C testbench: test vectors, autotb, xsim
# scripts), then xelab exactly as Vitis' run_xsim.sh would, then a *bounded* xsim run logging a VCD of the DUT ports;
# per-transform events and the RTL output data are read from the VCD. Needed because under 2023.2 the top-level
# ap_done of the final transform never fires (free-running auto-rewind processes in a dataflow region with start
# propagation disabled, HLS 200-656): a plain cosim_design ran to 31/32 transactions with all outputs produced and
# never terminated. Array configs (no_StagePipeline, original_C_style): plain cosim_design + VCD re-run.
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
ulimit -n 8192
E=/scratch/hc676/e2_hpfft; PY=/scratch/hc676/allo-agent/bin/python3
D=$E/$1/$2; cd $D || exit 1
for i in $(seq 1 720); do [ -f hls.done ] && break; sleep 20; done
[ -f hls.done ] || { echo "rc=98 no hls.done" > cosim.done; exit 1; }
grep -q "rc=0" hls.done || { echo "rc=97 csynth failed" > cosim.done; exit 1; }
[ -f testbench_orig.cpp ] || cp testbench.cpp testbench_orig.cpp
if grep -q "hls::stream" FFT.h; then cp $E/testbench_e2.cpp testbench.cpp; KIND=stream; NT=32; SETUP="-setup"; else cp $E/testbench_e2_array.cpp testbench.cpp; KIND=array; NT=3; SETUP=""; fi
[ -f e2_nt ] && { NT=$(cat e2_nt); sed -i "s/#define E2_NT [0-9]*/#define E2_NT $NT/" testbench.cpp; }
N=$(grep -oE "^#define FFT_NUM [0-9]+" FFT.h | awk '{print $3}'); UF=$(grep -oE "^#define UF [0-9]+" FFT.h | awk '{print $3}')
S=$D/build/FFT_300MHz
rm -rf $S/sim cosim.done cosim_events.json validate_*.json e2_rtl_*.txt
cat > cosim.tcl <<TCL
open_project build
open_solution FFT_300MHz
cosim_design $SETUP -trace_level port -rtl verilog
close_project
exit
TCL
T0=$(date +%s)
timeout -k 60 21600 vitis_hls -f cosim.tcl > cosim.log 2>&1; RC=$?
echo "rc=$RC cosim_wall_s=$(( $(date +%s) - T0 )) kind=$KIND N=$N UF=$UF NT=$NT method=$([ $KIND = stream ] && echo setup+bounded_xsim_vcd || echo cosim_design+vcd_rerun)" > cosim.done
[ -f $S/sim/wrapc/e2_outputs.txt ] && $PY $E/validate_fft.py $S/sim/wrapc/e2_inputs.txt $S/sim/wrapc/e2_outputs.txt $N $NT > validate_wrapc.json 2>&1
[ -f $S/sim/wrapc_pc/e2_outputs.txt ] && $PY $E/validate_fft.py $S/sim/wrapc_pc/e2_inputs.txt $S/sim/wrapc_pc/e2_outputs.txt $N $NT > validate_wrapc_pc.json 2>&1
cd $S/sim/verilog || { echo "vcd_rc=96 no sim dir" >> $D/cosim.done; exit 1; }
if [ $KIND = stream ]; then
  XELAB=$(grep -m1 "xelab" run_xsim.sh)
  T1=$(date +%s); eval "$XELAB" > e2_xelab.log 2>&1; XRC=$?
  echo "xelab_rc=$XRC xelab_wall_s=$(( $(date +%s) - T1 ))" >> $D/cosim.done
  [ $XRC -ne 0 ] && exit 1
fi
LAT=$($PY -c "import json,re;m=json.load(open('$S/FFT_300MHz_data.json'))['ModuleInfo']['Metrics']['FFT_TOP']['Latency'];print(max(int(x) for x in re.findall(r'\d+', m['LatencyWorst'])))")
II=$($PY -c "import json,re;m=json.load(open('$S/FFT_300MHz_data.json'))['ModuleInfo']['Metrics']['FFT_TOP']['Latency'];print(max([int(x) for x in re.findall(r'\d+', m['PipelineII'] or '')] or [0]))")
B=$(( N / (2*UF) )); TARGET=$(( NT * B ))
MAXNS=$(( (LAT + (NT+2)*II) * 3 * 3333 / 1000 + 20000 )); EXTRANS=$(( LAT * 3 * 3333 / 1000 + 5000 ))
cat > e2_vcd.tcl <<TCL
open_vcd e2.vcd
log_vcd [get_objects -filter {type == in_port || type == out_port || type == inout_port || type == port} /apatb_FFT_TOP_top/AESL_inst_FFT_TOP/*]
set t 0
set dc 0
set oc 0
set extra 0
while {\$t < $MAXNS} {
  run 2000 ns
  set t [expr \$t + 2000]
  catch { set dc [get_value -radix unsigned /apatb_FFT_TOP_top/done_cnt] }
  if {\$dc >= $NT} { run 1000 ns; break }
  if {\$dc >= $NT - 1} {
    # the last transform's ap_done never fires (HLS 200-656): allow 3 latencies more, then stop
    if {\$extra == 0} { set extra 1; run $EXTRANS ns; set t [expr \$t + $EXTRANS] ; catch { set dc [get_value -radix unsigned /apatb_FFT_TOP_top/done_cnt] }; break }
  }
}
puts "E2_SIM_STOP t_ns=\$t done_cnt=\$dc out_writes=\$oc bound_ns=$MAXNS"
close_vcd
quit
TCL
T2=$(date +%s)
timeout -k 60 14400 xsim --noieeewarnings FFT_TOP -tclbatch e2_vcd.tcl > e2_xsim.log 2>&1; SRC=$?
echo "xsim_rc=$SRC xsim_wall_s=$(( $(date +%s) - T2 )) $(grep -o 'E2_SIM_STOP.*' e2_xsim.log)" >> $D/cosim.done
[ -f e2.vcd ] && $PY $E/cosim_events.py e2.vcd $N $UF $NT --dump $D/e2_rtl > $D/cosim_events.json 2>&1
[ -s $D/e2_rtl_outputs.txt ] && $PY $E/validate_fft.py $D/e2_rtl_inputs.txt $D/e2_rtl_outputs.txt $N $NT > $D/validate_rtl_vcd.json 2>&1
exit 0
