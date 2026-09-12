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
cp "$ROOT"/run_spmw_micro.sh "$ROOT"/run_gem_stream.sh "$ROOT"/spmw_hier.sh "$STAGE/scripts/" 2>/dev/null

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
  grep -E "MXUVPU_STREAM|MXUVPU_TILE" "$ROOT/logs/gem_stream_S$S.log" \
    2>/dev/null > "$G/report/stream_cycles.txt"
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
