#!/bin/bash
# Directory queue: files queue/pending/<prio>_<name> hold "<kind> <size> <cfg>"; runs bash run_<kind>.sh
# with at most MAXJ concurrent jobs (own background jobs only -- no pgrep). Exits after 30 min idle or STOP.
E=/scratch/hc676/e2_hpfft
Q=${Q:-$E/queue}
MAXJ=${MAXJ:-4}
echo $$ > $Q/queue.pid
idle=0
while true; do
  running=$(jobs -rp | wc -l)
  f=$(ls $Q/pending 2>/dev/null | sort | head -1)
  if [ -n "$f" ] && [ "$running" -lt "$MAXJ" ]; then
    mv $Q/pending/$f $Q/running/$f
    read kind size cfg < $Q/running/$f
    echo "$(date +%FT%T) start $f: $kind $size $cfg"
    ( bash $E/run_${kind}.sh $size $cfg > $E/logs/${f}.log 2>&1; mv $Q/running/$f $Q/done/$f; echo "$(date +%FT%T) done $f" ) &
    idle=0; continue
  fi
  if [ -z "$f" ] && [ "$running" -eq 0 ]; then idle=$((idle+1)); [ $idle -ge 90 ] && break; fi
  [ -f $Q/STOP ] && break
  sleep 20
done
wait
echo "$(date +%FT%T) QUEUE_EXIT"
