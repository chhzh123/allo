#!/bin/bash
# Stage the E3 microbenchmark's sources, generated code and reports into the
# <framework>/<size>/{source,generated,report} shape E1, E2 and E4 use.
#
# Three frameworks, because three designs are compared and each has its own
# netlist: `spmw` is the programmable stage engine, `spmw-fixed` is the same
# workload on a fixed-function datapath, and `gemmini` is the baseline. Each
# framework's one-parameter ablation -- `noclip` for the programmable engine,
# `slice` for the fixed one -- lives in its parent's report/ rather than in a
# framework directory of its own, since it is the same source.
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

# The interior matrix cell, whichever role index it landed on. Hard-coding
# `mac_r4` was right for the programmable engine and wrong for anything else,
# so the role with the most instances is looked up instead.
interior () {                        # interior <build dir> <prefix>
  local top
  top=$(ls "$1/spmw_top.sv" "$1/sim/spmw_top.sv" 2>/dev/null | head -1)
  [ -n "$top" ] || return 1
  grep -oE "role ${2}_r[0-9]+: [0-9]+ instance" "$top" \
    | awk '{gsub(":","",$2); print $3, $2}' | sort -nr | head -1 | awk '{print $2}'
}

stage_spmw () {                      # stage_spmw <framework dir> <size> <tag> <source file>
  local fw=$1 S=$2 tag=$3 srcfile=$4
  local D=$STAGE/$fw/S$S
  local P=$ROOT/spmw_pnr${tag}_S$S   # the routed build: generated code and area
  mkdir -p "$D/source" "$D/generated" "$D/report"
  cp "$SRC/tests/dataflow/spmw/$srcfile" "$D/source/"

  for f in spmw_top.sv spmw_fifo.sv spmw_const.sv spmw_harness.sv; do
    cp "$P/$f" "$D/generated/" 2>/dev/null
  done
  for r in "$P"/*/; do
    local n; n=$(basename "$r")
    [ -f "$r/kernel.cpp" ] || continue
    cp "$r/kernel.cpp" "$D/generated/$n.cpp" 2>/dev/null
    cp "$r/$n.sv" "$D/generated/" 2>/dev/null
  done
  for f in util.rpt timing.rpt route.rpt util_hier.rpt util_synth.rpt cost.json; do
    cp "$P/$f" "$D/report/" 2>/dev/null
  done
  # One matrix cell and one vector lane's C synthesis, for the initiation
  # intervals: the numbers are quoted from these, not inferred from cycles.
  local mac vpu
  mac=$(interior "$P" mac); vpu=$(interior "$P" vpu)
  for r in $mac $vpu; do
    for f in "$P/$r/prj/sol/syn/report/"*_csynth.rpt; do
      [ -f "$f" ] && cp "$f" "$D/report/$(basename "$f")"
    done
  done
  echo "$mac $vpu" > "$D/report/roles_reported.txt"
  grep -hE "SPMW COSIM|SPMW CYCLES|SPMW XFORM" \
    "$ROOT/logs/spmw_cosim${tag}_S$S.log" 2>/dev/null | sed 's/^ *//' \
    | sort -u -k1,3 > "$D/report/cosim_cycles.txt"
  grep -E "SPMW STAGE|ARRAY WNS|SPMW UNROUTED|IMPLEMENTATION OK|array clock|HLS:" \
    "$ROOT/logs/spmw_pnr${tag}_S$S.log" 2>/dev/null > "$D/report/pnr_stages.txt"
}

for S in 4 8 16; do
  # ---- SPMW, the programmable stage engine ----------------------------------
  stage_spmw spmw "$S" "" test_spmw_tpu_micro.py
  # its ablation: the same netlist with the clip's five instructions dropped
  grep -hE "SPMW COSIM|SPMW CYCLES|SPMW XFORM" \
    "$ROOT/logs/spmw_cosim_noclip_S$S.log" 2>/dev/null | sed 's/^ *//' \
    | sort -u -k1,3 > "$STAGE/spmw/S$S/report/cosim_cycles_noclip.txt"

  # ---- SPMW, the fixed-function datapath ------------------------------------
  stage_spmw spmw-fixed "$S" "_fixed" test_spmw_tpu_micro_fixed.py
  # its ablation: the same design on SPMW's default depth-two register slices,
  # which is a different netlist, so it brings its own area and timing.
  F=$STAGE/spmw-fixed/S$S/report
  grep -hE "SPMW COSIM|SPMW CYCLES|SPMW XFORM" \
    "$ROOT/logs/spmw_cosim_slice_S$S.log" 2>/dev/null | sed 's/^ *//' \
    | sort -u -k1,3 > "$F/cosim_cycles_slice.txt"
  for f in util.rpt timing.rpt util_hier.rpt; do
    cp "$ROOT/spmw_pnr_slice_S$S/$f" "$F/slice_$f" 2>/dev/null
  done

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
