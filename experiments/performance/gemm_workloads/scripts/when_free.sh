#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# when_free.sh <GB> <command...>: start a command once the machine has that
# much memory available, and has had it for five minutes. The machine is
# shared, and a build that starts into the last of its memory takes someone
# else's with it.
main() {
  local need=$1; shift
  local held=0
  while [ $held -lt 5 ]; do
    if [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -ge "$need" ]; then
      held=$((held + 1))
    else
      held=0
    fi
    sleep 60
  done
  echo "$(date) start: $* (available $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) GB)" >> /scratch/hc676/e16_tiles/logs/lane.log
  exec "$@"
}
main "$@"
