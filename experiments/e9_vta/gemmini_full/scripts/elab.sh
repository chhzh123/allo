#!/bin/bash
# elab.sh <dim> [lean|shift|matmul|default]: elaborate a Rocket system with
# Gemmini as its RoCC accelerator and emit one SystemVerilog file per module.
# SIM=1 keeps the register initialisation a simulation needs; BUS=128 widens
# the system bus to Gemmini's DMA, as Chipyard does.
set -u
source /scratch/hc676/gemmini_env.sh
export PATH=/scratch/hc676/gemmini_full/bin:$PATH
export CHISEL_FIRTOOL_PATH=/scratch/hc676/toolchain/firtool/bin
export TMPDIR=/scratch/hc676/gemmini_full/tmp
export SBT_OPTS="-Xmx24G -Xss8M -Dsbt.ivy.home=/scratch/hc676/toolchain/ivy -Duser.home=/scratch/hc676/toolchain/home"
mkdir -p "$TMPDIR" /scratch/hc676/gemmini_full/logs
cd /scratch/hc676/gemmini_full || exit 2
D=$1; V=${2:-lean}
L=logs/elab_${V}_$D${BUS:+_b$BUS}${SIM:+_sim}.log
T0=$(date +%s)
echo "ELAB_START $V $D $(date)" > "$L"
MESH_DIM=$D GEMMINI_CONFIG=$V GEMMINI_SIM=${SIM:-0} GEMMINI_BUS=${BUS:-64} timeout 7200 sbt -batch "harness/runMain gen.ElaborateGemmini" >> "$L" 2>&1
echo "ELAB_RC=$? wall_s=$(( $(date +%s) - T0 ))" >> "$L"
