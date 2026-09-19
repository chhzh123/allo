#!/bin/bash
# Elaborate one VTA module to Verilog.  Own ivy home and user home so this
# does not disturb the Gemmini toolchain's caches, and TMPDIR off /tmp, which
# is full on this machine.
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH
command -v sbt >/dev/null || source /scratch/hc676/gemmini_env.sh 2>/dev/null
export TMPDIR=/scratch/hc676/vta_build/tmp
export SBT_OPTS="-Xmx12G -Dsbt.ivy.home=/scratch/hc676/vta_toolchain/ivy -Duser.home=/scratch/hc676/vta_toolchain/home"
mkdir -p "$TMPDIR" /scratch/hc676/vta_toolchain/ivy /scratch/hc676/vta_toolchain/home /scratch/hc676/vta_build/logs
TOP=${1:-TensorGemm}
L=/scratch/hc676/vta_build/logs/elab_$TOP.log
cd /scratch/hc676/vta/hardware/chisel || exit 2
T0=$(date +%s)
echo "VTA_ELAB_START $TOP $(date)" > "$L"
VTA_TOP=$TOP timeout 5400 sbt -batch "runMain gen.ElaborateVTA" >> "$L" 2>&1
echo "VTA_ELAB_RC=$? wall_s=$(( $(date +%s) - T0 ))" >> "$L"
