#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# collect_tiles.sh: stage exactly what reproduces E12's tables.
#   spmw_mesh/S<n>, spmw_kernel/S<n>        one build a size; a cycle record a
#                                           workload, report/W<side>_*.txt
#   <baseline>/S<n>_W<side>                 one build a workload
# each with source/, generated/ and report/. A baseline's generated kernel is
# megabytes and differs between workloads only in its loop bounds, so it is
# kept for the 64-cubed workload alone. Copies; touches nothing it reads.
set -u
E=/scratch/hc676/e16_tiles
T=/scratch/hc676/allo
STAGE=$E/stage
SIDES="64 128 256 512 1024"
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
for S in 16 32; do
  # ---- SPMW: the array alone -------------------------------------------------
  B=$E/spmw/mesh_S${S}_W64_pnr
  if [ -d $B ]; then
    D=$STAGE/spmw_mesh/S$S; mkdir -p $D/source $D/generated $D/report
    c $T/tests/dataflow/spmw/test_spmw_gemm_tiles.py $D/source/
    { echo "python3 scripts/spmw_build_array.py --design gemmtiles-<side> --size $S --frequency 300 --jobs 4 --cosim --out <dir>"
      echo "python3 scripts/spmw_build_array.py --design gemmtiles-64 --size $S --frequency 300 --jobs 4 --pnr --out <dir>"
    } > $D/source/command.txt
    for r in pe_r4 frame_r0; do c $B/$r/kernel.cpp $D/generated/$r.cpp; c $B/$r/$r.sv $D/generated/; done
    c $B/spmw_top.sv $B/spmw_fifo.sv $D/generated/
    c $B/pe_r4/prj/sol/syn/report/csynth.rpt $D/report/pe_r4_csynth.rpt
    c $B/frame_r0/prj/sol/syn/report/csynth.rpt $D/report/frame_r0_csynth.rpt
    for f in $B/pe_r4/prj/sol/syn/report/pe_r4_0_Pipeline_*_csynth.rpt; do c $f $D/report/; done
    c $B/util.rpt $B/timing.rpt $B/cost.json $D/report/
    c $E/spmw/mesh_S${S}_W64_pnr.log $D/report/build_pnr.log
    for W in $SIDES; do
      L=$E/spmw/mesh_S${S}_W${W}_cosim
      grep -hE "SPMW COSIM|SPMW CYCLES|^gemmtiles|^HLS:" $L.log > $D/report/W${W}_cosim.txt 2>/dev/null
      # One array for every workload: what each workload's build generated.
      [ -f $L/spmw_top.sv ] && same "W=$W roles+fabric" $L/pe_r*/kernel.cpp $L/frame_r0/kernel.cpp \
        $L/spmw_top.sv $L/spmw_fifo.sv >> $D/report/same_hardware.txt
    done
    same "pnr roles+fabric" $B/pe_r*/kernel.cpp $B/frame_r0/kernel.cpp $B/spmw_top.sv $B/spmw_fifo.sv \
      >> $D/report/same_hardware.txt
  fi
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
    for W in $SIDES; do for T in "" _beat2; do
      L=${K}_W$W$T
      [ -f $L.sim.log ] || continue
      { grep -hE "SPMW TB|xsim:|counts:|expected drain" $L.sim.log
        grep -m1 -oE "\.BEAT\([0-9]+\)" ${L}_sim/tb.sv 2>/dev/null | sed "s/^/memory /"
        P=$(grep -m1 -oE "always #[0-9.]+ ap_clk" ${L}_sim/tb.sv 2>/dev/null | grep -oE "[0-9.]+")
        [ -n "$P" ] && echo "clock period $(echo "$P" | awk '{print 2 * $1}') ns"
      } > $D/report/W${W}${T}_kernel_sim.txt
    done; done
    c ${K}_W64_sim/tb.sv $D/report/W64_tb.sv
  fi
  for W in $SIDES; do
    # ---- AutoSA ---------------------------------------------------------------
    # autosa_wide/        --host-serialize with K tiled, the way AutoSA's own
    #                     examples are run; the tile depth each workload's row
    #                     uses is in autosa_ktile.txt ("S W KT"), and the other
    #                     depths tried are kept as S<n>_W<side>_k<depth>
    # autosa/             no --host-serialize, K whole: the first runs
    # autosa_wide_whole/  --host-serialize with K whole, which fails
    for A in $E/autosa/narrow/S${S}_W$W $E/autosa/wide/S${S}_W$W $E/autosa/wide_k*/S${S}_W$W; do
      [ -d $A ] || continue
      V=$(basename $(dirname $A))
      case $V in
        narrow) D=$STAGE/autosa/S${S}_W$W;;
        wide)   D=$STAGE/autosa_wide_whole/S${S}_W$W;;
        *)      KT=${V#wide_k}
                D=$STAGE/autosa_wide/S${S}_W${W}_k$KT
                grep -qx "$S $W $KT" $E/autosa_ktile.txt 2>/dev/null && D=$STAGE/autosa_wide/S${S}_W$W;;
      esac
      mkdir -p $D/source $D/generated $D/report
      c $A/kernel.c $A/kernel.h $A/simd_info.json $A/hls/cosim.tcl $D/source/
      grep -h . $A/out/src/cmd > $D/source/autosa_command.txt 2>/dev/null
      if [ $W = 64 ]; then
        c $A/out/src/kernel_kernel.cpp $A/out/src/kernel_kernel.h $A/out/src/kernel_host.cpp $D/generated/
      else
        echo "As the 64-cubed build's generated/, with this workload's loop bounds; source/autosa_command.txt regenerates it." > $D/generated/README.txt
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
    done
    # AutoSA's own C testbench alone, on the serialized 64-cubed designs.
    for C in $E/keep/autosa_csim_S${S}_W${W}_k*; do
      [ -f $C/csim.log ] || continue
      KT=${C##*_k}
      for D in $STAGE/autosa_wide/S${S}_W${W}_k$KT $STAGE/autosa_wide/S${S}_W$W; do
        [ -d $D ] && grep -q "array_part\[$S,$S,$KT\]" $D/source/autosa_command.txt 2>/dev/null \
          && grep -hE "^Passed!|^Failed|CSIM (finish|start)|leftover" $C/csim.log > $D/report/csim_verdict.txt
      done
    done
    # ---- the Dato paper's array, at its appendix's depths and at the test's ----
    for KIND in paper test; do
      L=$E/dato/$KIND/S${S}_W$W
      [ -d $L ] || continue
      if [ $KIND = paper ]; then D=$STAGE/dato/S${S}_W$W; else D=$STAGE/dato_deep/S${S}_W$W; fi
      mkdir -p $D/source $D/generated $D/report
      c $L/gen/dato_mxu_${S}_$W.py $L/hls/tb.cpp $L/hls/csim.tcl $L/hls/run.tcl $D/source/
      if [ $W = 64 ]; then
        c $L/gen/kernel.cpp $L/gen/kernel.h $D/generated/
      else
        echo "As ../../S${S}_W64/generated, with this workload's loop bounds; scripts/dato_gen.py regenerates it." > $D/generated/README.txt
      fi
      R=$L/hls/prj/sol
      c $R/syn/report/top_csynth.rpt $R/sim/report/top_cosim.rpt $D/report/
      c $L/pnr/util.rpt $L/pnr/timing.rpt $L/pnr/util_hier.rpt $L/pnr/route_status.rpt $L/pnr/pnr.tcl $D/report/
      c $L/beats.txt $D/report/
      grep -v "LICENSE file" $L/gen.log > $D/report/gen.log 2>/dev/null
      width $R/syn/verilog/top.v > $D/report/port_widths.txt
      grep -hE "DATO TB" $L/hls/csim.log 2>/dev/null > $D/report/csim_verdict.txt
      grep -hE "DATO TB|co-simulation finished|^ERROR" $L/hls/build.log 2>/dev/null | head -12 > $D/report/cosim_verdict.txt
      c $E/logs/dato_${KIND}_S${S}_W$W.log $D/report/build.log
    done
    # The 64-cubed Dato design at the tool's default pipelining threshold, and
    # the test file's nested form at 128 cubed: what the tool made of each.
    U=$E/keep/dato_default_S${S}_W64/build.log
    if [ $W = 64 ] && [ -f $U ] && [ -d $STAGE/dato/S${S}_W64 ]; then
      { echo "tile loops the tool chose to pipeline: $(grep -c "HLS 214-376.*Loop< l_nt" $U), for example"
        grep -h -m2 "HLS 214-376.*Loop< l_nt" $U
        echo "loops unrolled completely by a factor of 64: $(grep -c "completely with a factor of 64" $U), for example"
        grep -h -m1 "completely with a factor of 64" $U
        grep -hE "XFORM 203-712|HLS 200-111\] Finished (Standard Transforms|Checking Synthesizability|Loop, function|Command)" $U
        cat $E/keep/dato_default_S${S}_W64/done.txt 2>/dev/null
      } > $STAGE/dato/S${S}_W64/report/default_threshold.txt
      c $E/keep/dato_default_S${S}_W64/syn.tcl $STAGE/dato/S${S}_W64/source/default_threshold.tcl
    fi
    if [ $W = 128 ] && [ -f $E/keep/dato_nested_S${S}_W128_messages.txt ] && [ -d $STAGE/dato/S${S}_W128 ]; then
      c $E/keep/dato_nested_S${S}_W128_messages.txt $STAGE/dato/S${S}_W128/report/nested_form_messages.txt
      c $E/keep/dato_nested_S${S}_W128_top_csynth.rpt $STAGE/dato/S${S}_W128/report/nested_form_top_csynth.rpt
    fi
    # ---- Allo's library array ----------------------------------------------------
    L=$E/allo/S${S}_W$W
    if [ -d $L ]; then
      D=$STAGE/allo/S${S}_W$W; mkdir -p $D/source $D/generated $D/report
      c $L/hls/tb.cpp $L/hls/tb_1launch.cpp $L/hls/csim.tcl $L/hls/run.tcl $D/source/
      echo "python3 experiments/e1_gemm/allo/S16/source/e1_allo_gen.py $S $W <dir>" > $D/source/command.txt
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
# ---- AutoSA's published int8 design, synthesised and simulated here ----------
for P in $E/autosa_paper/*; do
  [ -d $P ] || continue
  D=$STAGE/autosa_paper/$(basename $P); mkdir -p $D/source $D/report
  c $P/kernel.c $P/kernel.h $P/simd_info.json $P/hls/cosim.tcl $D/source/
  grep -h . $P/out/src/cmd > $D/source/autosa_command.txt 2>/dev/null
  R=$P/hls/cosim_prj/sol
  c $R/syn/report/kernel0_csynth.rpt $R/syn/report/PE_csynth.rpt $R/sim/report/kernel0_cosim.rpt $D/report/
  grep -hE "^Passed!|^Failed|co-simulation finished|^ERROR|leftover data" $P/hls/cosim.log 2>/dev/null | head -12 > $D/report/cosim_verdict.txt
  c $E/logs/autosa_paper_$(basename $P).log $D/report/build.log
done
# ---- the array's three forms at 4x4 and 8x8, 16 tiles each --------------------
D=$STAGE/spmw_forms; mkdir -p $D
for F in gemmtiles gemmtilesp gemmtiles2; do
  for P in "16 4" "32 8"; do
    set -- $P
    L=$E/spmw/probe_$F-$1_S$2
    [ -f $L.log ] || continue
    echo "$F S=$2 W=$1 | $(grep -hE "SPMW COSIM|SPMW CYCLES" $L.log | tr -s " " | tr "\n" "|") cell: $(grep -h "Pipelining result" $L/pe_r4/vitis_hls.log | head -n 1 | grep -oE "Final II = [0-9]+, Depth = [0-9]+")"
  done
done > $D/forms.txt
{ echo "The array as written (gemmtiles), with its cell left to pipeline (gemmtilesp)"
  echo "and with two-deep sum links (gemmtiles2): scripts/probe.sh <form>-<W> <S>."
  echo "Depth counts the loop's entry state: 2 is a one-state body."; } > $D/README.txt
mkdir -p $STAGE/scripts
c $E/spmw_tiles.sh $E/spmw_tiles_kernel.sh $E/spmw_all.sh $E/probe.sh $E/autosa_tiles.sh $E/autosa_tiles_tail.sh \
  $E/autosa_recosim.sh $E/autosa_paper.sh $E/autosa_ktile.txt $E/dato_tiles.sh $E/dato_gen.py $E/dato_tb.cpp \
  $E/allo_tiles.sh $E/lane.sh $E/when_free.sh $E/beats.py $E/collect_tiles.sh $STAGE/scripts/
find $STAGE -type f -empty -delete
du -sh $STAGE; find $STAGE -type f | wc -l
