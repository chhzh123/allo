#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# collect_tiles.sh: stage exactly what reproduces this folder's tables.
#   spmw_kernel/S<n>                        one build a size; a cycle record a
#                                           workload, report/W<side>_*.txt
#   autosa_wide/S<n>_W<side>, allo/S<n>_W<side>   one build a workload
# each with source/, generated/ and report/. A baseline's generated kernel is
# megabytes and differs between workloads only in its loop bounds, so it is
# kept for the 64-cubed workload alone. Copies; touches nothing it reads.
set -u
E=/scratch/hc676/e16_tiles
T=/scratch/hc676/allo
STAGE=$E/stage
SIDES="64 128 256 512"
rm -rf $STAGE
c() { cp "$@" 2>/dev/null; }   # a missing file is a missing result, not an error
width() {  # <rtl file>: the data width of each memory master, as synthesised
  grep -hoE "C_M_AXI_[A-Z0-9_]+_DATA_WIDTH *= *[0-9]+" "$1" 2>/dev/null | tr -s ' ' | sort -u
}
same() {  # <label> <file>...: one line saying what the files hash to together
  echo "$1 $(cat "${@:2}" 2>/dev/null | md5sum | cut -c1-32)"
}
monitor() {  # <cosim log> <dst>: what the simulation's deadlock monitor printed
  if grep -q "DEADLOCK DETECTED" $1 2>/dev/null; then
    { grep -h -m1 -B1 -A4 "DEADLOCK DETECTED" $1
      echo "reports: $(grep -c "Dependence cycle" $1)"
      grep "^// (1): Process:" $1 | sort | uniq -c
      grep -h "RTL Simulation :.*%" $1 | tail -n 1
      grep -h -m1 "finish called at time" $1 | cut -d: -f1-2
    } > $2
  fi
}
for S in 16; do
  # ---- SPMW: the same array as a kernel ---------------------------------------
  K=$E/spmw/kernel_S$S
  if [ -d $K ]; then
    D=$STAGE/spmw_kernel/S$S; mkdir -p $D/source $D/generated $D/report
    c $T/tests/dataflow/spmw/test_spmw_gemm_tiles.py $D/source/
    { echo "python3 scripts/spmw_package_kernel.py --design gemmtiles-256 --size $S --frequency 300 --jobs 4 --sim --out <kernel>"
      echo "python3 scripts/spmw_array_operands.py --design gemmtiles-<side> --size $S --out <operands>"
      echo "python3 scripts/spmw_kernel_sim.py <kernel> <operands> --design gemmtiles-<side> --size $S --polls 4000000"
      echo "vivado -mode batch -source pnr.tcl      # report/pnr.tcl"; } > $D/source/command.txt
    c $K/spmw_kernel.sv $K/spmw_top.sv $K/spmw_control_s_axi.sv $K/args.json $D/generated/
    for r in pe_r4 frame_r0; do c $K/roles/$r/kernel.cpp $D/generated/$r.cpp; done
    for f in $K/feeders/*; do c $f/kernel.cpp $D/generated/$(basename $f).cpp; done
    for f in $K/feeders/*; do
      c $f/prj/sol/syn/report/csynth.rpt $D/report/$(basename $f)_csynth.rpt
      width $f/prj/sol/syn/verilog/$(basename $f).v | sed "s/^/$(basename $f) /"
    done > $D/report/port_widths.txt
    c ${K}_pnr/util.rpt ${K}_pnr/timing.rpt ${K}_pnr/util_hier.rpt ${K}_pnr/route_status.rpt ${K}_pnr/pnr.tcl $D/report/
    c $K.package.log $D/report/package.log
    c $E/logs/spmw_kernel_S${S}_pnr.log $D/report/build.log
    same "kernel roles+fabric" $K/roles/pe_r*/kernel.cpp $K/roles/frame_r0/kernel.cpp $K/spmw_top.sv \
      $K/spmw_fifo.sv > $D/report/same_hardware.txt
    for W in $SIDES; do
      L=${K}_W$W
      [ -f $L.sim.log ] || continue
      { grep -hE "SPMW TB|xsim:|counts:|expected drain" $L.sim.log
        grep -m1 -oE "\.BEAT\([0-9]+\)" ${L}_sim/tb.sv 2>/dev/null | sed "s/^/memory /"
        P=$(grep -m1 -oE "always #[0-9.]+ ap_clk" ${L}_sim/tb.sv 2>/dev/null | grep -oE "[0-9.]+")
        [ -n "$P" ] && echo "clock period $(echo "$P" | awk '{print 2 * $1}') ns"
      } > $D/report/W${W}_kernel_sim.txt
    done
    c ${K}_W64_sim/tb.sv $D/report/W64_tb.sv
  fi
  for W in $SIDES; do
    # ---- AutoSA ---------------------------------------------------------------
    # autosa_wide/: --host-serialize with K tiled and latency hiding, the build
    # autosa_ktile.txt names for each workload ("S W <depth>_l<hiding>").
    KT=$(awk -v s=$S -v w=$W '$1 == s && $2 == w {print $3}' $E/autosa_ktile.txt 2>/dev/null)
    V=wide_k$KT; A=$E/autosa/$V/S${S}_W$W
    if [ -n "$KT" ] && [ -d $A ]; then
      D=$STAGE/autosa_wide/S${S}_W$W; mkdir -p $D/source $D/generated $D/report
      c $A/kernel.c $A/kernel.h $A/simd_info.json $A/hls/cosim.tcl $D/source/
      grep -h . $A/out/src/cmd > $D/source/autosa_command.txt 2>/dev/null
      # 390 KB a design: kept for the 64-cubed workload alone.
      if [ $W = 64 ]; then
        c $A/out/src/kernel_kernel.cpp $A/out/src/kernel_kernel.h $A/out/src/kernel_host.cpp $D/generated/
      else
        echo "source/autosa_command.txt regenerates it; ../../S${S}_W64/generated holds the 64-cubed build, whose K tile is 32 where this one's is 64." > $D/generated/README.txt
      fi
      R=$A/hls/cosim_prj/sol
      c $R/syn/report/kernel0_csynth.rpt $R/syn/report/PE_csynth.rpt $R/sim/report/kernel0_cosim.rpt $D/report/
      c $A/pnr/util.rpt $A/pnr/timing.rpt $A/pnr/util_hier.rpt $A/pnr/route_status.rpt $A/pnr/pnr.tcl $D/report/
      c $A/beats.txt $D/report/
      c $A/gen.log $D/report/autosa_gen.log
      width $R/syn/verilog/kernel0.v > $D/report/port_widths.txt
      for L in $A/hls/cosim_first.log $A/hls/cosim.log; do
        grep -hE "^Passed!|^Failed|co-simulation finished|^ERROR|leftover data" $L 2>/dev/null | head -12
      done > $D/report/cosim_verdict.txt
      monitor $A/hls/cosim.log $D/report/deadlock_monitor.txt
      c $E/logs/autosa_${V}_S${S}_W$W.log $D/report/build.log
    fi
    # ---- Allo's library array ----------------------------------------------------
    L=$E/allo/S${S}_W$W
    if [ -d $L ]; then
      D=$STAGE/allo/S${S}_W$W; mkdir -p $D/source $D/generated $D/report
      c $L/hls/tb.cpp $L/hls/tb_1launch.cpp $L/hls/csim.tcl $L/hls/run.tcl $D/source/
      echo "python3 experiments/performance/gemm_arrays/allo/S16/source/e1_allo_gen.py $S $W <dir>" > $D/source/command.txt
      if [ $W = 64 ]; then
        c $L/gen/kernel.cpp $D/generated/
      else
        echo "As ../../S${S}_W64/generated, with this workload's loop bounds; source/command.txt regenerates it." > $D/generated/README.txt
      fi
      c $L/gen/e1_gen_info.json $D/generated/
      R=$L/hls/prj/sol
      c $R/syn/report/gemm_csynth.rpt $R/sim/report/gemm_cosim.rpt $D/report/
      c $L/pnr/util.rpt $L/pnr/timing.rpt $L/pnr/util_hier.rpt $L/pnr/route_status.rpt $L/pnr/pnr.tcl $D/report/
      c $L/beats.txt $D/report/
      grep -E "^E1 GEN (OK|ERROR)" $L/gen.log > $D/report/gen.log 2>/dev/null
      width $R/syn/verilog/gemm.v > $D/report/port_widths.txt
      grep -hE "E1 TB" $L/hls/csim.log 2>/dev/null > $D/report/csim_verdict.txt
      grep -hE "E1 TB|co-simulation finished|^ERROR" $L/hls/build.log 2>/dev/null | head -12 > $D/report/cosim_verdict.txt
      c $E/logs/allo_S${S}_W$W.log $D/report/build.log
    fi
  done
done
mkdir -p $STAGE/scripts
c $E/spmw_tiles_kernel.sh $E/autosa_tiles2.sh $E/autosa_ktile.txt $E/allo_tiles.sh $E/beats.py \
  $E/lane.sh $E/stagger.sh $E/when_free.sh $E/collect_tiles.sh $STAGE/scripts/
find $STAGE -type f -empty -delete
du -sh $STAGE; find $STAGE -type f | wc -l
