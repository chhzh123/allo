#!/bin/bash
# Allo's cosimulation can hang indefinitely in Vitis's deadlock-monitor step,
# "Generating RTL test bench". A flat no-growth timer cannot tell that from a
# healthy csynth, which is legitimately silent for many minutes on a loaded
# machine -- an earlier version of this script killed three good builds that
# way. So the stall test is signature-based: only the known hang line, quiet,
# is a stall. Anything else quiet is just slow, and gets a long ceiling.
set -u
S=$1
LOG=/scratch/hc676/e1_allo_beats/S$S/build.log
HANG_RE="Generating RTL test bench"
QUIET_HANG=600      # 10 min quiet *at the hang signature* is a hang
QUIET_OTHER=3600    # 60 min quiet anywhere else is a giving-up ceiling
for attempt in 1 2 3; do
  echo "=== S=$S attempt $attempt ==="
  setsid bash /scratch/hc676/allo_beats.sh $S > /scratch/hc676/allo_run_S$S.log 2>&1 &
  runner=$!
  quiet=0
  while kill -0 $runner 2>/dev/null; do
    before=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    sleep 120
    after=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    if [ "$before" = "$after" ] && [ "$after" != "0" ]; then
      quiet=$((quiet + 120))
      tailline=$(tail -1 "$LOG" 2>/dev/null)
      limit=$QUIET_OTHER
      case "$tailline" in *"$HANG_RE"*) limit=$QUIET_HANG;; esac
      if [ $quiet -ge $limit ]; then
        echo "  S=$S stalled after ${quiet}s quiet at: ${tailline:0:70}"
        for p in $(pgrep -x vitis_hls); do
          case "$(readlink /proc/$p/cwd 2>/dev/null)" in
            */e1_allo_beats/S$S*) kill -9 $p 2>/dev/null;;
          esac
        done
        kill -9 $runner 2>/dev/null
        break
      fi
    else
      quiet=0
    fi
  done
  wait $runner 2>/dev/null
  cat /scratch/hc676/allo_run_S$S.log
  if grep -qE "^S=$S rc=0 " /scratch/hc676/allo_run_S$S.log 2>/dev/null; then
    bash /scratch/hc676/vcd_allo.sh $S 2>&1 | tail -6
    break
  fi
done
echo "ALLO_RETRY_DONE_$S"
