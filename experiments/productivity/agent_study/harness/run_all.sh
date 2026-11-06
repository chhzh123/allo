#!/bin/bash
# The fifteen graded trials: five models, three arms each.
# One model at a time, its three arms concurrently, so the three arms of a
# paired comparison see identical machine conditions.
set -u
S=/scratch/hc676/agentstudy
export TRIAL_TOKENS=${TRIAL_TOKENS:-500000}
export TRIAL_ROUTES=${TRIAL_ROUTES:-3}
export TRIAL_BUILDS=${TRIAL_BUILDS:-8}
MODELS=${MODELS:-"anthropic/claude-opus-5 openai/gpt-5.6-sol moonshotai/kimi-k3 z-ai/glm-5.3 deepseek/deepseek-v4-pro"}
mkdir -p "$S/trials"
for MODEL in $MODELS; do
  TAG=$(echo "$MODEL" | tr '/.' '__')
  echo "===== launching $MODEL  $(date +%FT%T)  load $(cut -d' ' -f1 /proc/loadavg)"
  for ARM in spmw hls rtl; do
    bash "$S/harness/run_trial.sh" "$ARM" "$MODEL" "${TAG}__${ARM}" \
      > "$S/trials/${TAG}__${ARM}.log" 2>&1 &
    sleep 20
  done
done
wait
for MODEL in $MODELS; do
  TAG=$(echo "$MODEL" | tr '/.' '__')
  echo "===== $MODEL"
  for ARM in spmw hls rtl; do
    printf '  %-5s ' "$ARM"
    python3 -c "
import json, sys
try:
    s = json.load(open('$S/trials/${TAG}__'+'$ARM'+'/summary.json'))
except Exception as e:
    print('no summary:', e); raise SystemExit
print('submitted=%s tokens=%s builds=%s model_s=%s tool_s=%s'
      % (s['submitted'], s['tokens'], s['builds'], s['model_seconds'], s['tool_seconds']))
" 2>&1 | tail -1
  done
done
echo "ALL_TRIALS_DONE $(date +%FT%T)"
