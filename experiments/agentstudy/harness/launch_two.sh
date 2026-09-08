#!/bin/bash
# Opus 5 and GPT-5.6 Sol, three arms each, on the fixed harness.
set -u
cd /scratch/hc676/agentstudy || exit 2
export TRIAL_TOKENS=500000 TRIAL_BUILDS=8 TRIAL_ROUTES=3
for MODEL in anthropic/claude-opus-5 openai/gpt-5.6-sol; do
  TAG=$(echo "$MODEL" | tr '/.' '__')
  for ARM in spmw hls rtl; do
    setsid nohup bash harness/run_trial.sh "$ARM" "$MODEL" "${TAG}__${ARM}" \
      > "trials/${TAG}__${ARM}.log" 2>&1 < /dev/null &
    sleep 20
  done
done
echo "six trials launched"
date +%T
