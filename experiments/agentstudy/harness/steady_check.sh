#!/bin/bash
# Re-measure the interval-8 designs over twelve products instead of two.
set -u
S=/scratch/hc676/agentstudy
for n in openai_gpt-5_6-sol__rtl anthropic_claude-opus-5__rtl deepseek_deepseek-v4-pro__rtl deepseek_deepseek-v4-pro__hls; do
  arm=${n##*__}
  T=$S/steady/$n; rm -rf "$T"; mkdir -p "$T"; cp "$S/trials/$n"/design.* "$T"/ 2>/dev/null || cp "$S/designs/$n"/design.* "$T"/ 2>/dev/null
  echo "===== $n"
  case $arm in
    rtl)  timeout 1800 bash "$S/arms/rtl/build.sh" "$T" steady 2>&1 | grep -E "STUDY VALUES|STUDY RESULT|STUDY INTERVAL" | head -14 ;;
    hls)  timeout 3600 bash "$S/arms/hls/build.sh" "$T" steady 2>&1 | grep -E "STUDY VALUES|STUDY RESULT|STUDY INTERVAL" | head -14 ;;
  esac
done
echo STEADY_DONE
