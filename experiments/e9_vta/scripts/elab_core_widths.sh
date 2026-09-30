#!/bin/bash
# Elaborate VTA Core at widths 4 and 8, one after the other (one sbt project).
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH
command -v sbt >/dev/null || source /scratch/hc676/gemmini_env.sh 2>/dev/null
export TMPDIR=/scratch/hc676/vta_build/tmp
export SBT_OPTS="-Xmx12G -Dsbt.ivy.home=/scratch/hc676/vta_toolchain/ivy -Duser.home=/scratch/hc676/vta_toolchain/home"
cd /scratch/hc676/vta/hardware/chisel || exit 2
for w in 4 8; do
  L=/scratch/hc676/vta_build/logs/elab_Core_w$w.log
  T0=$(date +%s)
  VTA_TOP=Core VTA_WIDTH=$w timeout 5400 sbt -batch "runMain gen.ElaborateVTA" > "$L" 2>&1
  echo "VTA_ELAB_RC=$? wall_s=$(( $(date +%s) - T0 ))" >> "$L"
done
