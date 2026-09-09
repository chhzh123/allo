#!/bin/bash
# Race-safe worker over /scratch/hc676/e2_hpfft/queue_pnr/pending (files "<kind> <size> <cfg>", run via
# run_<kind>.sh). A job is claimed by a successful mv into running/ (only the winner proceeds). Concurrency is a
# heavy-job budget of 2 for this package minus the previous agent's still-running P&R queue (pid 2128793, until its
# log says QUEUE_EXIT) minus the serial cosim worker (flag file cosim_worker.active). Exits when pending is empty
# and nothing runs, or on STOP2.
E=/scratch/hc676/e2_hpfft; Q=$E/queue_pnr
echo $$ > $Q/worker.pid
budget() {
  local b=2
  if [ -d /proc/2128793 ] && ! grep -q QUEUE_EXIT $E/logs/queue_pnr.log 2>/dev/null; then b=$((b-1)); fi
  [ -f $E/cosim_worker.active ] && b=$((b-1))
  echo $b
}
while true; do
  [ -f $Q/STOP2 ] && { echo "$(date +%FT%T) STOP2 seen"; break; }
  running=$(jobs -rp | wc -l); bud=$(budget)
  f=$(ls $Q/pending 2>/dev/null | sort | head -1)
  if [ -n "$f" ] && [ "$running" -lt "$bud" ]; then
    if mv $Q/pending/$f $Q/running/$f 2>/dev/null; then
      read kind size cfg < $Q/running/$f
      echo "$(date +%FT%T) start $f: $kind $size $cfg (running=$running budget=$bud)"
      ( bash $E/run_${kind}.sh $size $cfg > $E/logs/${f}.log 2>&1; mv $Q/running/$f $Q/done/$f; echo "$(date +%FT%T) done $f" ) &
      sleep 5; continue
    fi
  fi
  if [ -z "$f" ] && [ "$running" -eq 0 ]; then break; fi
  sleep 30
done
wait
echo "$(date +%FT%T) WORKER_EXIT"
