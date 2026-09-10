#!/bin/bash
# Standalone Gemmini MXU+VPU: the mesh plus the accumulator scale/activation
# path, no scratchpad, controller, DMA or RoCC. Scope is scale + ReLU;
# has_normalizations=false gates LAYERNORM/IGELU/SOFTMAX off, so Normalizer is
# present only for the types AccumulatorScaleIO refers to.
set -u
SRC=/scratch/hc676/gemmini_src/src/main/scala/gemmini
OLD=/scratch/hc676/gemmini_build
NEW=/scratch/hc676/gemmini_mxuvpu

rm -rf "$NEW"; mkdir -p "$NEW"
cp "$OLD/build.sbt" "$NEW/"
cp -a "$OLD/src" "$NEW/"                       # gemmini array files + hardfloat + gen
rm -f "$NEW/src/main/scala/gemmini/MeshDriver.scala" "$NEW/src/main/scala/gen/"*.scala
for f in AccumulatorMem AccumulatorScale Activation Normalizer NormCmd Pipeline SharedExtMem; do
  cp "$SRC/$f.scala" "$NEW/src/main/scala/gemmini/" && echo "  added $f.scala"
done
echo "SETUP_DONE"; ls "$NEW/src/main/scala/gemmini" | tr '\n' ' '
