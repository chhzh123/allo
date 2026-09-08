#!/bin/bash
# Re-run the matched cosim snapshot with a waveform and read off the same two
# events the SPMW testbench reports.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1
SIM=/scratch/hc676/e1_autosa_matched/hls/S$S/cosim_prj/sol/sim/verilog
[ -f $SIM/run_xsim.sh ] || { echo "S=$S no cosim snapshot"; exit 1; }
cd $SIM || exit 1
XELAB=$(grep -m1 "bin/xelab" run_xsim.sh | sed "s/-s kernel0 *$/-s kernel0_fo -debug typical/")
bash -c "$XELAB" > fo_xelab.log 2>&1 || { echo "S=$S xelab failed"; tail -3 fo_xelab.log; exit 1; }
D=/apatb_kernel0_top/AESL_inst_kernel0
{ echo "open_vcd kernel0_fo.vcd"
  for s in ap_clk m_axi_gmem_A_RVALID m_axi_gmem_A_RREADY m_axi_gmem_B_RVALID \
           m_axi_gmem_B_RREADY m_axi_gmem_C_WVALID m_axi_gmem_C_WREADY; do
    echo "log_vcd $D/$s"
  done
  echo "run all"; echo "close_vcd"; echo "quit"; } > fo_vcd.tcl
xsim --noieeewarnings kernel0_fo -tclbatch fo_vcd.tcl > fo_xsim.log 2>&1
printf "S=%-3s " $S
python3 /scratch/hc676/beats.py kernel0_fo.vcd 2>&1 | tail -1
