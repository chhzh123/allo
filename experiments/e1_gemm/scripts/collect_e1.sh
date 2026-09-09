#!/bin/bash
# Stage exactly what reproduces the matched E1 table, one shape for every
# framework:  <framework>/<size>/{source,generated,report}
set -u
STAGE=/scratch/hc676/e1_stage
rm -rf $STAGE
for S in 4 8 16 32; do
  # ---- SPMW: the mesh (array scope) and the kernel (memory scope) ----------
  for V in mesh kernel; do
    D=$STAGE/spmw_$V/S$S; mkdir -p $D/source $D/generated $D/report
    if [ $V = mesh ]; then B=/scratch/hc676/e1_spmw_${S}_pnr; C=/scratch/hc676/e1_spmw_${S}_cosim
    else B=/scratch/hc676/e1_spmwmem_${S}_pnr; C=/scratch/hc676/e1_memsim_${S}; fi
    # source: the design and the command that built it
    cp /scratch/hc676/allo/tests/dataflow/spmw/test_spmw_gemm_int8.py $D/source/ 2>/dev/null
    [ $V = kernel ] && cp /scratch/hc676/allo/tests/dataflow/spmw/test_spmw_autosa_match.py $D/source/ 2>/dev/null
    # generated: one role's C++ and wrapper, and the assembled fabric
    for r in $(ls -d $B/pe_r0 $B/feed_r0 2>/dev/null | head -1); do
      cp $r/kernel.cpp $D/generated/$(basename $r).cpp 2>/dev/null
      cp $r/$(basename $r).sv $D/generated/ 2>/dev/null
      cp $r/prj/sol/syn/report/csynth.rpt $D/report/$(basename $r)_csynth.rpt 2>/dev/null
    done
    cp $B/spmw_top.sv $D/generated/ 2>/dev/null
    cp $B/spmw_fifo.sv $D/generated/ 2>/dev/null
    # report: routed resources and timing, and the run that gave the cycles
    cp $B/util.rpt $B/timing.rpt $B/cost.json $D/report/ 2>/dev/null
    cp /scratch/hc676/e1_spmw_${S}_pnr.log $D/report/build_pnr.log 2>/dev/null
    if [ $V = mesh ]; then
      cp /scratch/hc676/e1_spmw_${S}_cosim.log $D/report/cosim.log 2>/dev/null
      grep -hE "SPMW COSIM|SPMW CYCLES" $C/sim/xsim.log > $D/report/cycles.txt 2>/dev/null
    else
      cp /scratch/hc676/e1_memsim_${S}.sim.log $D/report/kernel_sim.log 2>/dev/null
      grep -hE "SPMW TB" /scratch/hc676/e1_memsim_${S}.sim.log > $D/report/cycles.txt 2>/dev/null
      cp /scratch/hc676/e1_memsim_${S}.package.log $D/report/package.log 2>/dev/null
    fi
  done
  # ---- AutoSA, matched -----------------------------------------------------
  A=/scratch/hc676/e1_autosa_matched
  if [ -d $A/S$S ]; then
    D=$STAGE/autosa/S$S; mkdir -p $D/source $D/generated $D/report
    cp $A/S$S/kernel.c $A/S$S/kernel.h $D/source/ 2>/dev/null
    grep -h . $A/S$S/out/src/cmd > $D/source/autosa_command.txt 2>/dev/null
    cp $A/S$S/out/src/kernel_kernel.cpp $A/S$S/out/src/kernel_kernel.h $D/generated/ 2>/dev/null
    R=$A/hls/S$S/cosim_prj/sol
    cp $R/syn/report/kernel0_csynth.rpt $R/syn/report/PE_csynth.rpt $D/report/ 2>/dev/null
    cp $R/sim/report/kernel0_cosim.rpt $D/report/ 2>/dev/null
    cp $A/pnr/S$S/util.rpt $A/pnr/S$S/timing.rpt $D/report/ 2>/dev/null
    cp $A/hls_S$S.log $D/report/build_hls.log 2>/dev/null
    cp $A/pnr_S$S.log $D/report/build_pnr.log 2>/dev/null
  fi
  # ---- Allo, the tile workload (already S*S*S) -----------------------------
  L=/scratch/hc676/e1_allo_w1/S${S}_tile
  if [ -d $L ]; then
    D=$STAGE/allo/S$S; mkdir -p $D/source $D/generated $D/report
    cp /scratch/hc676/e1_allo_w1/drv/e1_allo_gen.py $D/source/ 2>/dev/null
    cp /scratch/hc676/allo/allo/library/systolic.py $D/source/allo_library_systolic.py 2>/dev/null
    cp $L/gemm.mlir $D/generated/ 2>/dev/null
    cp $L/kernel.cpp $D/generated/ 2>/dev/null
    cp $L/out.prj/solution1/syn/report/gemm_csynth.rpt $D/report/ 2>/dev/null
    cp $L/out.prj/solution1/syn/report/PE_kernel_gemm_0_0_csynth.rpt $D/report/ 2>/dev/null
    ls $L/out.prj/solution1/sim/report/*.rpt >/dev/null 2>&1 && cp $L/out.prj/solution1/sim/report/*.rpt $D/report/ 2>/dev/null
    for f in $L/out.prj/solution1/impl/report/verilog/*.rpt; do cp "$f" $D/report/ 2>/dev/null; done
    cp $L/stages.txt $D/report/ 2>/dev/null
  fi
done
echo "=== staged:"; du -sh $STAGE; find $STAGE -maxdepth 2 -type d | sort | sed "s|$STAGE/||" | head -30
