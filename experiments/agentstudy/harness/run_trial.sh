#!/bin/bash
# One trial, end to end. usage: run_trial.sh <arm> <model-slug> <trial-name>
# Reads the key from ~/.openrouter_key, exports it to the agent only, and never
# prints it. The key file is expected to be mode 600 and outside the repository.
set -u
ARM=$1; MODEL=$2; NAME=$3
S=/scratch/hc676/agentstudy
KEYFILE=${OPENROUTER_KEY_FILE:-$HOME/.openrouter_key}
[ -r "$KEYFILE" ] || { echo "no readable key file at $KEYFILE"; exit 2; }
perms=$(stat -c %a "$KEYFILE")
[ "$perms" = "600" ] || { echo "key file is mode $perms; run chmod 600 $KEYFILE"; exit 2; }
export OPENROUTER_API_KEY="$(cat "$KEYFILE")"
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
T=$S/trials/$NAME; rm -rf "$T"; mkdir -p "$T"
cd /scratch/hc676/allo
python3 -u "$S/harness/agent.py" --arm "$ARM" --model "$MODEL" --trial "$T" \
  --tokens "${TRIAL_TOKENS:-200000}" --builds "${TRIAL_BUILDS:-40}" 2>&1 | tail -5
echo "TRIAL_DONE $NAME"
