#!/bin/bash
# e4_job.sh <log> <cmd...>: run detached, log to <log>, write <log>.done at the end
LOG=$1; shift
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
cd /scratch/hc676/e4_work
rm -f $LOG.done
setsid nohup bash -c "( $* ) > $LOG 2>&1; echo rc=\$? >> $LOG; touch $LOG.done" > /dev/null 2>&1 < /dev/null &
echo "launched: $LOG"
