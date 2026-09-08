#!/bin/bash
# Both GLM trials again, this time with reasoning bounded.
set -u
cd /scratch/hc676/agentstudy || exit 2
mkdir -p repeated
for n in z-ai_glm-5_3__hls z-ai_glm-5_3__spmw; do
  for suffix in "" .transcript.jsonl .summary.json .log; do
    [ -e "trials/$n$suffix" ] && mv "trials/$n$suffix" "repeated/${n}_unbounded$suffix" 2>/dev/null
  done
done
export TRIAL_TOKENS=500000 TRIAL_BUILDS=8 TRIAL_ROUTES=3
setsid nohup bash harness/run_trial.sh spmw z-ai/glm-5.3 z-ai_glm-5_3__spmw > trials/z-ai_glm-5_3__spmw.log 2>&1 < /dev/null &
sleep 15
setsid nohup bash harness/run_trial.sh hls z-ai/glm-5.3 z-ai_glm-5_3__hls > trials/z-ai_glm-5_3__hls.log 2>&1 < /dev/null &
sleep 5
echo "both GLM trials relaunched with reasoning bounded"
date +%T
