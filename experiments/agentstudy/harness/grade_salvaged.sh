#!/bin/bash
# Grade the nine designs the models wrote before my bug killed their trials.
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
# No API cost: the artifacts are on disk. This checks the grading pipeline on
# real model output and previews what the study will find.
set -u
S=/scratch/hc676/agentstudy
OUT=$S/salvage; mkdir -p $OUT
for d in $S/discarded_launch_3/*__*/; do
  name=$(basename "$d")
  arm=${name##*__}
  [ -d "$d" ] || continue
  ls "$d"/design.* >/dev/null 2>&1 || continue
  T=$OUT/$name; rm -rf "$T"; mkdir -p "$T"; cp "$d"/design.* "$T"/
  echo "===== $name"
  case $arm in
    spmw) timeout 3600 bash "$S/arms/spmw/build.sh" "$T" visible 2>&1 | grep -vE "^ *(pe|stage|reorder)[a-z0-9_]*: rc=" | tail -12 ;;
    hls)  timeout 3600 bash "$S/arms/hls/build.sh" "$T" visible 2>&1 | tail -12 ;;
    rtl)  timeout 1800 bash "$S/arms/rtl/build.sh" "$T" visible 2>&1 | tail -12 ;;
  esac
done
echo SALVAGE_DONE
