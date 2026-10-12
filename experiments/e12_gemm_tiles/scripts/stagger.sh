#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# stagger.sh <GB> <gap seconds> <command>...: start the commands one after
# another, each once the machine has had that much memory available for three
# minutes, and the given gap apart, so the front end of one build has shown
# what it takes before the next begins. The machine is shared.
main() {
  local need=$1 gap=$2; shift 2
  local cmd held avail
  cd /scratch/hc676/e16_tiles || exit 1
  for cmd in "$@"; do
    held=0
    while [ $held -lt 3 ]; do
      avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
      if [ "$avail" -ge "$need" ]; then held=$((held + 1)); else held=0; fi
      sleep 60
    done
    echo "$(date) stagger start: $cmd (available $avail GB)" >> logs/lane.log
    setsid nohup bash -c "$cmd" > /dev/null 2>&1 < /dev/null &
    sleep "$gap"
  done
  echo "$(date) stagger: all $# started" >> logs/lane.log
}
main "$@"
