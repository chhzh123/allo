#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# reroute_dir.sh <build dir> <period ns> <out dir>: place and route a built
# array again at another clock target, from the same HLS output.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
B=$1; P=$2; R=$3
rm -rf "$R"; mkdir -p "$R"; cd "$R" || exit 2
echo "create_clock -period $P -name ap_clk [get_ports ap_clk]" > clock.xdc
sed "s#$B/clock.xdc#$R/clock.xdc#" "$B/assemble.tcl" > assemble.tcl
T0=$(date +%s)
vivado -mode batch -source assemble.tcl -nojournal -log vivado.log > /dev/null 2>&1
WNS=$(grep "^ARRAY WNS" vivado.log | awk '{print $3}')
echo "REROUTE build=$B target=$P wns=$WNS unrouted=$(grep '^SPMW UNROUTED' vivado.log | awk '{print $3}') wall_s=$(( $(date +%s) - T0 ))" | tee result.txt
