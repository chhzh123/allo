#!/bin/bash
# Stage the E3 microbenchmark's sources, generated code and reports into the
# <framework>/<size>/{source,generated,report} shape E1, E2 and E4 use.
#
# Run on brg-zhang-xcel; rsync $STAGE back into the repo afterwards.
set -u
ROOT=/scratch/hc676/e3_micro
STAGE=${1:-/scratch/hc676/e3_micro/stage}
SRC=$ROOT/allo
rm -rf "$STAGE"; mkdir -p "$STAGE/scripts" "$STAGE/stimulus"

cp "$ROOT"/stim/stim_S*.txt "$STAGE/stimulus/"
for f in run_spmw_micro.sh run_gem_stream.sh run_gem_xsim.sh spmw_hier.sh \
         elab_mxuvpu.sh gen_mxuvpu_tb.py collect_micro.py; do
  cp "$ROOT/$f" "$STAGE/scripts/" 2>/dev/null
done

for S in 4 8 16; do
  # ---- SPMW -----------------------------------------------------------------
  D=$STAGE/spmw/S$S
  mkdir -p "$D/source" "$D/generated" "$D/report"
  cp "$SRC/tests/dataflow/spmw/test_spmw_tpu_micro.py" "$D/source/"

  P=$ROOT/spmw_pnr_S$S             # the routed build: generated code and area
  C=$ROOT/spmw_cosim_S$S           # the cosimulated build: cycles
  for f in spmw_top.sv spmw_fifo.sv spmw_const.sv spmw_harness.sv; do
    cp "$P/$f" "$D/generated/" 2>/dev/null
  done
  for r in "$P"/*/; do
    n=$(basename "$r")
    [ -f "$r/kernel.cpp" ] || continue
    cp "$r/kernel.cpp" "$D/generated/$n.cpp" 2>/dev/null
    cp "$r/$n.sv" "$D/generated/" 2>/dev/null
  done

  for f in util.rpt timing.rpt route.rpt util_hier.rpt util_synth.rpt cost.json; do
    cp "$P/$f" "$D/report/" 2>/dev/null
  done
  # One matrix cell and one vector lane's C synthesis, for the initiation
  # intervals: the numbers are quoted from these, not inferred from cycles.
  for r in mac_r4 vpu_r1; do
    for f in "$P/$r/prj/sol/syn/report/"*_csynth.rpt; do
      [ -f "$f" ] && cp "$f" "$D/report/$(basename "$f")"
    done
  done
  grep -hE "SPMW COSIM|SPMW CYCLES|SPMW XFORM" "$ROOT/logs/spmw_cosim_S$S.log" \
    2>/dev/null | sed 's/^ *//' | sort -u -k1,3 > "$D/report/cosim_cycles.txt"
  grep -hE "SPMW COSIM|SPMW CYCLES|SPMW XFORM" \
    "$ROOT/logs/spmw_cosim_noclip_S$S.log" 2>/dev/null | sed 's/^ *//' \
    | sort -u -k1,3 > "$D/report/cosim_cycles_noclip.txt"
  grep -E "SPMW STAGE|ARRAY WNS|SPMW UNROUTED|IMPLEMENTATION OK|array clock|HLS:" \
    "$ROOT/logs/spmw_pnr_S$S.log" 2>/dev/null > "$D/report/pnr_stages.txt"

  # ---- Gemmini --------------------------------------------------------------
  G=$STAGE/gemmini/S$S
  mkdir -p "$G/source" "$G/generated" "$G/report"
  cp "$ROOT/gemmini/src/main/scala/gen/MxuVpuStream.scala" "$G/source/" 2>/dev/null
  cp "$ROOT/gemmini/src/main/scala/gen/MxuVpu.scala" "$G/source/" 2>/dev/null
  # Both harnesses, kept apart: xsim is what the table quotes and chiseltest
  # is the cross-check, and mixing them in one file would hide which is which.
  grep -E "MXUVPU_STREAM |MXUVPU_TILE" "$ROOT/logs/gem_xsim_S$S.log" \
    2>/dev/null > "$G/report/xsim_cycles.txt"
  grep -E "MXUVPU_STREAM |MXUVPU_TILE" "$ROOT/logs/gem_stream_S$S.log" \
    2>/dev/null > "$G/report/chiseltest_cycles.txt"
  [ -s "$G/report/chiseltest_cycles.txt" ] || cat > "$G/report/chiseltest_cycles.txt" <<TXT
# chiseltest did not finish at this size.
#
# With no verilator on the machine chiseltest falls back to a Scala
# interpreter: 63s at 4x4, 25 minutes at 8x8, and E1 measured six hours for a
# 16x16 mesh. The run was retired once the xsim harness -- calibrated against
# chiseltest at 4x4 and 8x8, where the two differ by a constant one cycle of
# interval -- had covered this size in 16 seconds.
TXT
  cp "$ROOT/xsim_S$S/tb.sv" "$G/generated/tb_mxuvpu.sv" 2>/dev/null
  cat > "$G/generated/README.md" <<MD
# Elaborated Verilog is not committed here

\`MxuVpu.v\` at S=$S is megabytes of Chisel output -- 2.4 MB at 16x16 -- and it
is one command away:

    cd <gemmini project>
    MESH_DIM=$S SCALE_MODE=shift sbt -batch "runMain gen.ElaborateMxuVpu"

which writes \`mxuvpu_out_${S}_shift/MxuVpu.v\`. That is the netlist
\`../../../gemmini/source/mxuvpu_pnr.sh\` routes and
\`../../../gemmini/report/S$S/\` reports on.
MD
done
du -sh "$STAGE"; find "$STAGE" -type f | wc -l
echo STAGE_DONE
