#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# reroute.sh <size> <period ns>: place and route a built engine again at
# another clock target, from the same HLS output, into route_<size>_p<period>.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1; P=$2
B=/scratch/hc676/e13_mem/b_ptpumem-micro_$S
R=/scratch/hc676/e13_mem/route_${S}_p${P/./}
rm -rf "$R"; mkdir -p "$R"; cd "$R" || exit 2
echo "create_clock -period $P -name ap_clk [get_ports ap_clk]" > clock.xdc
sed "s#$B/clock.xdc#$R/clock.xdc#" "$B/assemble.tcl" > assemble.tcl
T0=$(date +%s)
vivado -mode batch -source assemble.tcl -nojournal -log vivado.log > /dev/null 2>&1
WNS=$(grep "^ARRAY WNS" vivado.log | awk '{print $3}')
echo "REROUTE size=$S target=$P wns=$WNS unrouted=$(grep '^SPMW UNROUTED' vivado.log | awk '{print $3}') wall_s=$(( $(date +%s) - T0 ))" | tee result.txt
