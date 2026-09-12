#!/bin/bash
# One size of the Gemmini streaming microbenchmark, against the shared stimulus.
#
# Each size gets its own copy of the project: two `sbt -batch` runs in one
# directory share `target/` and step on each other's compilation.
#
# Six hours, not two. E1 measured its 16x16 mesh driver at six hours under the
# same backend -- chiseltest's default is a Scala interpreter and there is no
# verilator on this machine -- and a 5400s cap there produced empty cells.
set -u
source /scratch/hc676/gemmini_env.sh
export SBT_OPTS="-Xmx12G -Dsbt.ivy.home=/scratch/hc676/toolchain/ivy -Duser.home=/scratch/hc676/toolchain/home"
S=$1
SRC=/scratch/hc676/e3_micro/gemmini
DIR=/scratch/hc676/e3_micro/gemmini_S${S}
LOG=/scratch/hc676/e3_micro/logs/gem_stream_S${S}.log
mkdir -p /scratch/hc676/e3_micro/logs
rm -f "$LOG.done"
[ -d "$DIR" ] || rsync -a --exclude test_run_dir "$SRC/" "$DIR/"
rsync -a "$SRC/src/" "$DIR/src/"
cd "$DIR" || exit 2
T0=$(date +%s)
STIM=/scratch/hc676/e3_micro/stim/stim_S${S}.txt SCALE_MODE=shift \
  timeout 21600 sbt -batch "runMain gen.MxuVpuStreamDriver" > "$LOG" 2>&1
echo "rc=$? wall_s=$(( $(date +%s) - T0 ))" >> "$LOG"
touch "$LOG.done"
