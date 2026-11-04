#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_tapa_collect.sh: stage the E12 AutoSA (TAPA) builds in the shape of
# experiments/e12_gemm_tiles/<baseline>/S16_W<side>/{source,generated,report}
# under export/autosa_tapa/, plus the drivers' logs under export/autosa_tapa/logs.
# generated/ is kept for 64^3 and 512^3 (K tiled by 32 and by 64); 128^3 and
# 256^3 differ from 512^3's only in loop bounds and get a note.
set -u
E=/scratch/hc676/e22_autosa_tapa/e12
X=$E/export/autosa_tapa
rm -rf $X; mkdir -p $X/logs
cp $E/driver.log $E/compile_times.txt $E/cosim_results.txt $E/pnr_results.txt $X/logs/
for W in 64 128 256 512; do
  G=$E/gen/W$W B=$E/build/W$W C=$E/cosim/W$W P=$E/pnr/W$W D=$X/S16_W$W
  mkdir -p $D/source $D/generated $D/report
  cp $G/kernel.c $G/kernel.h $G/simd_info.json $G/sa_sizes.txt $D/source/
  sed "s|$G|<build>|g" $G/out/src/cmd > $D/source/autosa_command.txt
  if [ $W = 64 ] || [ $W = 512 ]; then
    cp $G/out/src/kernel_kernel.cpp $G/out/src/kernel_kernel.h \
       $G/out/src/kernel_host.cpp $G/out/src/kernel_host.h $D/generated/
  else
    n=$(diff $G/out/src/kernel_kernel.cpp $E/gen/W512/out/src/kernel_kernel.cpp | grep -c '^[<>]')
    echo "kernel_kernel.cpp differs from ../../S16_W512/generated/kernel_kernel.cpp in $n lines, its loop bounds; regenerate with scripts/autosa_tapa_gen.sh $W 64" > $D/generated/README.txt
  fi
  cp $G/out/src/swsim.log $D/report/swsim.log
  cp $B/compile.log $D/report/compile.log
  for m in A_IO_L2_in A_IO_L2_in_boundary B_IO_L2_in B_IO_L2_in_boundary \
           C_drain_IO_L1_out_wrapper C_drain_IO_L1_out_boundary_wrapper PE_wrapper PE; do
    cp $B/work/report/${m}_csynth.rpt $D/report/
  done
  cp $C/cosim.log $D/report/cosim.log
  v=$(grep -h -o 'Passed!\|Failed with [0-9]* errors' $C/cosim.log | head -1)
  { echo "$v"; [ "$v" = 'Passed!' ] && echo "TAPA RTL simulation: PASS (the host's check against AutoSA's golden model)"; } > $D/report/cosim_verdict.txt
  grep -h 'BEATS' $C/cosim.log | head -1 > $D/report/beats.txt
  grep -oE '(input|output) \[[0-9]+:0\] m_axi_[ABC]_(RDATA|WDATA)' $B/work/hdl/kernel0.v \
    | awk '{split($2,a,/[\[:]/); p=$3; sub(/^m_axi_/,"",p); sub(/_(R|W)DATA$/,"",p);
            if (!(p in w)) { w[p]=a[2]+1; order[++n]=p } }
           END { for (i=1;i<=n;i++) printf "C_M_AXI_%s_DATA_WIDTH = %d\n", order[i], w[order[i]] }' \
    > $D/report/port_widths.txt
  cp $P/pnr.tcl $P/util.rpt $P/util_hier.rpt $P/timing.rpt $P/route_status.rpt $D/report/
done
du -sh $X
