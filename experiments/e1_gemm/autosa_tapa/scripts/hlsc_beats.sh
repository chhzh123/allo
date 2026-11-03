#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# hlsc_beats.sh <S>: rebuild E1's wide AutoSA design (Vitis HLS C target) with E1's recipe
# (experiments/e1_gemm/scripts/hls_wide.sh) in a fresh directory, then replay its cosim
# with a VCD and print beats.py's per-port census (scripts/vcd_matched.sh's method).
main() {
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
local S=$1 G=/scratch/hc676/e1_autosa_wide/S$1/out/src H=/scratch/hc676/e22_autosa_tapa/hlsc/S$1
rm -rf $H; mkdir -p $H; cd $H
cp $G/kernel_kernel.cpp $G/kernel_kernel.h $G/kernel_host.cpp /scratch/hc676/e1_autosa_wide/S$S/kernel.h .
local SIG PB CB DA DC
SIG=$(grep -oE "void kernel0\([^)]*" kernel_kernel.cpp | head -1)
PB=$(echo "$SIG" | grep -oE "A_t[0-9]+" | tr -dc 0-9); CB=$(echo "$SIG" | grep -oE "C_t[0-9]+" | tr -dc 0-9)
DA=$(( S*S/PB )); [ $DA -lt 1 ] && DA=1; DC=$(( S*S/CB )); [ $DC -lt 1 ] && DC=1
sed -i -e "s/\(#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem_A\)[[:space:]]*$/\1 depth=$DA/" \
       -e "s/\(#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem_B\)[[:space:]]*$/\1 depth=$DA/" \
       -e "s/\(#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem_C\)[[:space:]]*$/\1 depth=$DC/" kernel_kernel.cpp
printf "open_project cosim_prj\nset_top kernel0\nadd_files kernel_kernel.cpp\nadd_files -tb kernel_host.cpp\nopen_solution sol\nset_part xcu280-fsvh2892-2L-e\ncreate_clock -period 3.333 -name default\ncsynth_design\ncosim_design -rtl verilog\nexit\n" > cosim.tcl
vitis_hls -f cosim.tcl > cosim.log 2>&1
grep -E "^Passed!|^Failed with" cosim.log | head -1
local SIM=$H/cosim_prj/sol/sim/verilog
cd $SIM || return 1
local XELAB=$(grep -m1 "bin/xelab" run_xsim.sh | sed "s/-s kernel0 *$/-s kernel0_e22 -debug typical/")
bash -c "$XELAB" > e22_xelab.log 2>&1 || { echo "xelab failed"; return 1; }
local D=/apatb_kernel0_top/AESL_inst_kernel0 s
{ echo "open_vcd $H/k.vcd"
  for s in ap_clk m_axi_gmem_A_RVALID m_axi_gmem_A_RREADY m_axi_gmem_B_RVALID m_axi_gmem_B_RREADY \
           m_axi_gmem_C_WVALID m_axi_gmem_C_WREADY m_axi_gmem_A_ARVALID m_axi_gmem_A_ARREADY; do
    echo "log_vcd $D/$s"; done
  echo "run all"; echo "close_vcd"; echo "quit"; } > $H/vcd.tcl
xsim --noieeewarnings kernel0_e22 -tclbatch $H/vcd.tcl > $H/xsim.log 2>&1
python3 /scratch/hc676/beats.py $H/k.vcd A:m_axi_gmem_A_RVALID:m_axi_gmem_A_RREADY \
  B:m_axi_gmem_B_RVALID:m_axi_gmem_B_RREADY C:m_axi_gmem_C_WVALID:m_axi_gmem_C_WREADY \
  AR:m_axi_gmem_A_ARVALID:m_axi_gmem_A_ARREADY
rm -f $H/k.vcd
}
main "$@"
