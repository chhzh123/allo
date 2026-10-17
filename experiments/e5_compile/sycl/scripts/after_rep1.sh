#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# after_rep1.sh: once the first repetition of all four sizes is timed, stop the
# campaign before it starts another 32x32 compile -- one takes eight hours or
# more -- and time repetitions 2 and 3 of 4x4, 8x8 and 16x16 alone.
main() {
  set -u
  local E=/scratch/hc676/e17_sycl pid a c
  until [ $(( $(wc -l < $E/runs/results.csv) - 1 )) -ge 4 ]; do sleep 60; done
  for pid in $(ps -u hc676 -o pid=); do
    a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$a" in "/bin/bash ./sycl_compile.sh 3 4 8 16 32 ") kill $pid;; esac
  done
  sleep 5
  for pid in $(ps -u hc676 -o pid=); do
    c=$(readlink /proc/$pid/cwd 2>/dev/null)
    case "$c" in $E/runs/S*_r2|$E/runs/S*_r2/*) kill $pid 2>/dev/null;; esac
  done
  sleep 10
  rm -rf $E/runs/S*_r2
  echo "$(date -Iseconds) after_rep1: first campaign stopped after repetition 1; repetitions 2-3 of 4 8 16" >> $E/runs/driver.out
  cd $E && exec ./sycl_compile.sh 2-3 4 8 16
}
main "$@"
