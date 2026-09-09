#!/bin/bash
set -u
cd /scratch/hc676/agentstudy || exit 2
for suffix in "" .transcript.jsonl .summary.json .log; do
  [ -e "trials/z-ai_glm-5_3__rtl$suffix" ] && \
    mv "trials/z-ai_glm-5_3__rtl$suffix" "repeated/z-ai_glm-5_3__rtl_unbounded$suffix"
done
export TRIAL_TOKENS=500000 TRIAL_BUILDS=8 TRIAL_ROUTES=3
setsid nohup bash harness/run_trial.sh rtl z-ai/glm-5.3 z-ai_glm-5_3__rtl \
  > trials/z-ai_glm-5_3__rtl.log 2>&1 < /dev/null &
sleep 5
echo "glm SystemVerilog repeated under the bound"
