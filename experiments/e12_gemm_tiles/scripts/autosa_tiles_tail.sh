#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_tiles_tail.sh <S> <W> <narrow|wide> <vitis_hls pid, or 0>: finish
# autosa_tiles.sh's flow for a build whose driver was stopped while its Vitis
# HLS run and its route were still going -- steps 2's verdict and 3 of that
# script, unchanged, then the route's lines. Needed once: the driver was
# edited while six builds were running from it, and a shell reads its script
# as it goes.
main() {
  set -u
  export PATH=/scratch/hc676/allo-agent/bin:$PATH
  source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
  S=$1; W=$2; V=$3; PID=$4
  E=/scratch/hc676/e16_tiles
  D=$E/autosa/$V/S${S}_W$W
  H=$D/hls
  LOG=$E/logs/autosa_${V}_S${S}_W$W.log
  START=$(stat -c %Y $H/cosim.tcl)
  echo "tail: driver replaced by autosa_tiles_tail.sh $(date)" >> $LOG
  if [ $PID != 0 ]; then
    while kill -0 $PID 2>/dev/null; do sleep 30; done
    grep -q "co-simulation finished: PASS" $H/cosim.log; RC=$?
    echo "hls rc=$RC wall_s=$(( $(date +%s) - START ))" >> $LOG
    grep -E "^Passed!|^Failed with|C/RTL co-simulation finished|DEADLOCK DETECTED" $H/cosim.log | head -4 >> $LOG
    grep -E "\| *kernel0 *\|" $H/cosim_prj/sol/sim/report/*_cosim.rpt 2>/dev/null | head -3 | cut -c1-140 >> $LOG
    # ---- 3. first memory beat to last ----------------------------------------
    SIM=$H/cosim_prj/sol/sim/verilog
    if [ ! -f $SIM/run_xsim.sh ]; then
      echo "beats: no cosim snapshot" >> $LOG
    elif [ $W -gt ${BEATS_MAX:-256} ]; then
      echo "beats: not rerun at this size" >> $LOG
    else
      cd $SIM
      XELAB=$(grep -m1 "bin/xelab" run_xsim.sh | sed "s/-s kernel0 *$/-s kernel0_fo -debug typical/")
      bash -c "$XELAB" > fo_xelab.log 2>&1 || echo "beats xelab failed" >> $LOG
      T=/apatb_kernel0_top/AESL_inst_kernel0
      { echo "open_vcd kernel0_fo.vcd"
        for s in ap_clk m_axi_gmem_A_RVALID m_axi_gmem_A_RREADY m_axi_gmem_B_RVALID \
                 m_axi_gmem_B_RREADY m_axi_gmem_C_WVALID m_axi_gmem_C_WREADY; do
          echo "log_vcd $T/$s"
        done
        echo "run all"; echo "close_vcd"; echo "quit"; } > fo_vcd.tcl
      xsim --noieeewarnings kernel0_fo -tclbatch fo_vcd.tcl > fo_xsim.log 2>&1
      python3 $E/beats.py kernel0_fo.vcd > $D/beats.txt 2>&1
      cat $D/beats.txt >> $LOG
      rm -f kernel0_fo.vcd
    fi
  fi
  # The route was started by the driver and runs on: wait for its last line.
  until grep -q "^pnr LUT" $D/pnr_lines.txt 2>/dev/null; do
    grep -q "Finished Command csynth_design" $H/cosim.log $H/cosim_first.log 2>/dev/null || break
    sleep 30
  done
  if [ -f $D/pnr_lines.txt ]; then cat $D/pnr_lines.txt >> $LOG; else echo "pnr: no RTL" >> $LOG; fi
  echo "AUTOSA_TILES_DONE $(date)" >> $LOG
}
main "$@"
