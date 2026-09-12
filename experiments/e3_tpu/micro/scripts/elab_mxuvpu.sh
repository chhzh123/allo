#!/bin/bash
# Emit MxuVpu's Verilog at each size, for an xsim testbench. chiseltest falls
# back to a Scala interpreter here (no verilator on the machine) and it scales
# badly with mesh size; xsim on the emitted RTL does not.
set -u
source /scratch/hc676/gemmini_env.sh
export SBT_OPTS="-Xmx8G -Dsbt.ivy.home=/scratch/hc676/toolchain/ivy -Duser.home=/scratch/hc676/toolchain/home"
DIR=/scratch/hc676/e3_micro/gemmini_elab
LOG=/scratch/hc676/e3_micro/logs/elab.log
rm -f "$LOG.done"
[ -d "$DIR" ] || rsync -a --exclude test_run_dir /scratch/hc676/e3_micro/gemmini/ "$DIR/"
cd "$DIR" || exit 2
: > "$LOG"
for S in 4 8 16; do
  MESH_DIM=$S SCALE_MODE=shift timeout 3600 sbt -batch "runMain gen.ElaborateMxuVpu" >> "$LOG" 2>&1
  echo "S=$S rc=$? $(ls -la $DIR/mxuvpu_out_${S}_shift/MxuVpu.v 2>/dev/null | awk '{print $5}')" >> "$LOG"
done
touch "$LOG.done"
