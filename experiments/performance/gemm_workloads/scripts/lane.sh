#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# lane.sh <driver> <S> <max> <W>...: run `<driver> <S> <W> [LANE_ARGS]` for
# each W in turn, never more than <max> of that driver at that size at once,
# and never starting one with less than LANE_MIN_GB (default 100) of memory
# available. A 32x32 build peaks at 40 GB in its first minutes and holds 14
# to 29 for hours at its end; started together, five of them and their routes
# ask for more than the machine has, and it is not only ours.
set -u
DRIVER=$1; S=$2; MAX=$3; shift 3
E=/scratch/hc676/e16_tiles; cd $E
running() {  # builds of this kind and size now running, by their workload
  ps -u $(id -un) -o args= | grep "^/bin/bash ./$DRIVER $S " | awk '{print $4}' | sort -u | wc -l
}
for W in "$@"; do
  while :; do
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    busy=$(running)
    [ $busy -lt $MAX ] && [ $avail -ge ${LANE_MIN_GB:-100} ] && break
    sleep 120
  done
  echo "$(date) start $DRIVER $S $W ${LANE_ARGS:-} (available ${avail} GB, running $busy)" >> $E/logs/lane.log
  nohup ./$DRIVER $S $W ${LANE_ARGS:-} > /dev/null 2>&1 &
  sleep ${LANE_GAP:-1500}
done
echo "$(date) lane $DRIVER $S done" >> $E/logs/lane.log
