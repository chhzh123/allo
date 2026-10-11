#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# collect_ws.sh: stage exactly what reproduces the weight-stationary table, one
# shape for every framework: <framework>/S<size>/{source,generated,report}.
# Copies; touches nothing it reads.
set -u
E=/scratch/hc676/e15_ws
T=/scratch/hc676/allo
STAGE=$E/stage
rm -rf $STAGE
c() { cp "$@" 2>/dev/null; }   # a missing file is a missing result, not an error
width() {  # <rtl file>: the data width of each memory master, as synthesised
  grep -hoE "C_M_AXI_[A-Z0-9_]+_DATA_WIDTH *= *[0-9]+" "$1" 2>/dev/null | tr -s ' ' | sort -u
}
for S in 4 8 16 32; do
  # ---- SPMW: the mesh alone ------------------------------------------------
  D=$STAGE/spmw_mesh/S$S; mkdir -p $D/source $D/generated $D/report
  B=$E/spmw/mesh_S${S}_pnr; C=$E/spmw/mesh_S${S}_cosim
  c $T/tests/dataflow/spmw/test_spmw_gemm_ws.py $D/source/
  for mode in cosim pnr; do
    echo "python3 scripts/spmw_build_array.py --design gemmws --size $S --frequency 300 --jobs 4 --$mode --out <dir>"
  done > $D/source/command.txt
  c $B/mac_r4/kernel.cpp $D/generated/mac_r4.cpp
  c $B/mac_r4/mac_r4.sv $B/spmw_top.sv $B/spmw_fifo.sv $D/generated/
  c $B/mac_r4/prj/sol/syn/report/csynth.rpt $D/report/mac_r4_csynth.rpt
  for f in $B/mac_r4/prj/sol/syn/report/mac_r4_0_Pipeline_*_csynth.rpt; do c $f $D/report/; done
  c $B/util.rpt $B/timing.rpt $B/cost.json $D/report/
  c $E/spmw/mesh_S${S}_pnr.log $D/report/build_pnr.log
  c $E/spmw/mesh_S${S}_cosim.log $D/report/build_cosim.log
  c $C/sim/xsim.log $D/report/xsim.log
  grep -hE "SPMW COSIM|SPMW CYCLES" $E/spmw/mesh_S${S}_cosim.log > $D/report/cycles.txt 2>/dev/null
  # ---- SPMW: the same mesh as a kernel -------------------------------------
  D=$STAGE/spmw_kernel/S$S; mkdir -p $D/source $D/generated $D/report
  K=$E/spmw/kernel_S$S
  c $T/tests/dataflow/spmw/test_spmw_gemm_ws.py $D/source/
  { echo "python3 scripts/spmw_package_kernel.py --design gemmws --size $S --frequency 300 --jobs 4 --sim --out <kernel>"
    echo "python3 scripts/spmw_array_operands.py --design gemmws --size $S --out <operands>"
    echo "python3 scripts/spmw_kernel_sim.py <kernel> <operands> --design gemmws --size $S"
    echo "vivado -mode batch -source pnr.tcl      # report/pnr.tcl"; } > $D/source/command.txt
  c $K/spmw_kernel.sv $K/spmw_top.sv $K/spmw_control_s_axi.sv $K/args.json $D/generated/
  c $K/roles/mac_r4/kernel.cpp $D/generated/mac_r4.cpp
  for f in $K/feeders/*; do c $f/kernel.cpp $D/generated/$(basename $f).cpp; done
  for f in $K/feeders/*; do
    c $f/prj/sol/syn/report/csynth.rpt $D/report/$(basename $f)_csynth.rpt
    width $f/prj/sol/syn/verilog/$(basename $f).v | sed "s/^/$(basename $f) /"
  done > $D/report/port_widths.txt
  c ${K}_pnr/util.rpt ${K}_pnr/timing.rpt ${K}_pnr/util_hier.rpt ${K}_pnr/route_status.rpt ${K}_pnr/pnr.tcl $D/report/
  c $K.package.log $D/report/package.log
  c $K.sim.log $D/report/kernel_sim.log
  c ${K}_sim/xsim.log $D/report/xsim.log
  c ${K}_sim/tb.sv $D/report/tb.sv
  grep -hE "SPMW TB" $K.sim.log > $D/report/cycles.txt 2>/dev/null
  c $E/logs/spmw_kernel_S$S.log $D/report/build.log
  # ---- AutoSA, as generated and with --host-serialize ----------------------
  for V in narrow wide; do
    A=$E/autosa/$V/S$S
    [ -d $A ] || continue
    if [ $V = narrow ]; then D=$STAGE/autosa/S$S; else D=$STAGE/autosa_wide/S$S; fi
    mkdir -p $D/source $D/generated $D/report
    c $A/kernel.c $A/kernel.h $A/simd_info.json $D/source/
    grep -h . $A/out/src/cmd > $D/source/autosa_command.txt 2>/dev/null
    c $A/out/src/kernel_kernel.cpp $A/out/src/kernel_kernel.h $A/out/src/kernel_host.cpp $D/generated/
    c $A/hls/cosim.tcl $D/source/
    R=$A/hls/cosim_prj/sol
    c $R/syn/report/kernel0_csynth.rpt $R/syn/report/PE_csynth.rpt $R/sim/report/kernel0_cosim.rpt $D/report/
    c $A/pnr/util.rpt $A/pnr/timing.rpt $A/pnr/util_hier.rpt $A/pnr/route_status.rpt $A/pnr/pnr.tcl $D/report/
    c $A/beats.txt $D/report/
    c $A/gen.log $D/report/autosa_gen.log
    width $R/syn/verilog/kernel0.v > $D/report/port_widths.txt
    grep -hE "^Passed!|^Failed|co-simulation finished|^ERROR" $A/hls/cosim.log 2>/dev/null | head -12 > $D/report/cosim_verdict.txt
    c $E/logs/autosa_${V}_S$S.log $D/report/build.log
  done
  # ---- Allo ----------------------------------------------------------------
  L=$E/allo/S$S
  if [ -d $L ]; then
    D=$STAGE/allo/S$S; mkdir -p $D/source $D/generated $D/report
    c $E/ws_allo_gen.py $L/gen/tb.cpp $L/gen/tb_1launch.cpp $L/hls/csim.tcl $L/hls/run.tcl $D/source/
    c $L/gen/kernel.cpp $L/gen/ws_gen_info.json $D/generated/
    R=$L/hls/prj/sol
    c $R/syn/report/gemm_csynth.rpt $R/sim/report/gemm_cosim.rpt $D/report/
    c $R/syn/report/PE_kernel_ws_gemm_0_0_csynth.rpt $R/syn/report/PE_kernel_ws_gemm_1_1_csynth.rpt $D/report/
    for f in $R/syn/report/PE_kernel_ws_gemm_1_1_Pipeline_*_csynth.rpt; do c $f $D/report/; done
    c $L/pnr/util.rpt $L/pnr/timing.rpt $L/pnr/util_hier.rpt $L/pnr/route_status.rpt $L/pnr/pnr.tcl $D/report/
    c $L/beats.txt $D/report/
    c $L/gen.log $D/report/gen.log
    width $R/syn/verilog/gemm.v > $D/report/port_widths.txt
    grep -hE "WS TB" $L/hls/csim.log 2>/dev/null > $D/report/csim_verdict.txt
    grep -hE "WS TB|co-simulation finished|^ERROR" $L/hls/build.log 2>/dev/null | head -12 > $D/report/cosim_verdict.txt
    c $E/logs/allo_S$S.log $D/report/build.log
  fi
done
mkdir -p $STAGE/scripts
c $E/autosa_ws.sh $E/allo_ws.sh $E/ws_allo_gen.py $E/spmw_ws.sh $E/spmw_ws_kernel.sh $E/beats.py $E/collect_ws.sh $E/collect_e1_full.sh $STAGE/scripts/
find $STAGE -type f -empty -delete
du -sh $STAGE; find $STAGE -type f | wc -l
