#!/bin/bash
# Every finished design, measured over twelve products rather than two.
set -u
S=/scratch/hc676/agentstudy
for d in $S/trials/*__*/; do
  n=$(basename "$d"); arm=${n##*__}
  ls "$d"design.* >/dev/null 2>&1 || continue
  T=$S/steady/$n; rm -rf "$T"; mkdir -p "$T"; cp "$d"design.* "$T"/
  echo "===== $n"
  case $arm in
    spmw) timeout 3600 bash "$S/arms/spmw/build.sh" "$T" steady 2>&1 | grep -E "STUDY VALUES|STUDY RESULT|STUDY PRODUCT 0 |STUDY INTERVAL|STUDY BUILD" | head -18 ;;
    hls)  timeout 3600 bash "$S/arms/hls/build.sh"  "$T" steady 2>&1 | grep -E "STUDY VALUES|STUDY RESULT|STUDY PRODUCT 0 |STUDY INTERVAL|STUDY BUILD" | head -18 ;;
    rtl)  timeout 1800 bash "$S/arms/rtl/build.sh"  "$T" steady 2>&1 | grep -E "STUDY VALUES|STUDY RESULT|STUDY PRODUCT 0 |STUDY INTERVAL|STUDY BUILD" | head -18 ;;
  esac
done
echo STEADY_ALL_DONE
