#!/bin/bash
# Read Allo's memory beats off its own cosim snapshot, the same two events the
# SPMW testbench reports and the same rising-edge convention.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1
SIM=/scratch/hc676/e1_allo_beats/S$S/prj/sol/sim/verilog
[ -f $SIM/run_xsim.sh ] || { echo "S=$S no cosim snapshot"; exit 1; }
cd $SIM || exit 1
XELAB=$(grep -m1 "bin/xelab" run_xsim.sh | sed "s/-s gemm *$/-s gemm_fo -debug typical/")
bash -c "$XELAB" > fo_xelab.log 2>&1 || { echo "S=$S xelab failed"; tail -3 fo_xelab.log; exit 1; }
D=/apatb_gemm_top/AESL_inst_gemm
{ echo "open_vcd gemm_fo.vcd"
  echo "log_vcd $D/ap_clk"
  for g in 0 1 2; do for s in RVALID RREADY WVALID WREADY; do echo "log_vcd $D/m_axi_gmem${g}_$s"; done; done
  echo "run all"; echo "close_vcd"; echo "quit"; } > fo_vcd.tcl
xsim --noieeewarnings gemm_fo -tclbatch fo_vcd.tcl > fo_xsim.log 2>&1
printf "Allo S=%-3s\n" $S
python3 /scratch/hc676/beats.py gemm_fo.vcd \
  A:m_axi_gmem0_RVALID:m_axi_gmem0_RREADY \
  B:m_axi_gmem1_RVALID:m_axi_gmem1_RREADY \
  C:m_axi_gmem2_WVALID:m_axi_gmem2_WREADY
