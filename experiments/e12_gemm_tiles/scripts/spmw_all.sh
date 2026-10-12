#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# spmw_all.sh: every SPMW run of E12, in an order that keeps the long ones
# from asking for memory together: the routes, the kernel packages, the three
# forms at 4x4 and 8x8 and the four shorter workloads first; the kernel's
# simulations once it is packaged; the 1024-cubed launches last, the two
# kernel ones twenty minutes apart (each holds 19 GB while it prepares).
cd /scratch/hc676/e16_tiles
for S in 16 32; do
  ./spmw_tiles.sh $S 64 pnr 6 &
  ./spmw_tiles_kernel.sh $S package 6 &
done
for d in gemmtiles gemmtilesp gemmtiles2; do
  ./probe.sh $d-16 4 3 &
  ./probe.sh $d-32 8 3 &
done
for S in 16 32; do for W in 64 128 256 512; do ./spmw_tiles.sh $S $W cosim 3 & done; done
sleep 60
until grep -q DONE logs/spmw_kernel_S16_package.log && grep -q DONE logs/spmw_kernel_S32_package.log; do sleep 30; done
for S in 16 32; do
  ./spmw_tiles_kernel.sh $S pnr &
  for W in 64 128 256 512; do ./spmw_tiles_kernel.sh $S sim $W & done
done
for S in 16 32; do ./spmw_tiles.sh $S 1024 cosim 3 & done
./spmw_tiles_kernel.sh 32 sim 1024 &
sleep 1200
./spmw_tiles_kernel.sh 16 sim 1024 &
wait
echo "SPMW_ALL_DONE $(date)" >> logs/spmw_all.log
