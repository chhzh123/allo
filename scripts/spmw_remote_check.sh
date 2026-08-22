#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sync this tree to brg-zhang-xcel and run the SPMW suite there.
#
# SPMW is pure Python, so no rebuild is needed; the remote's editable install
# picks the changes up directly. Vitis is sourced only when asked for, since the
# simulator levels do not need it.
#
# Usage:
#   bash scripts/spmw_remote_check.sh              # sync + the whole SPMW suite
#   bash scripts/spmw_remote_check.sh gemm         # one file (tests/dataflow/spmw/test_spmw_gemm.py)
#   HLS=1 bash scripts/spmw_remote_check.sh        # also source the Vitis env

set -euo pipefail

HOST="${HOST:-brg-zhang-xcel}"
REMOTE_DIR="${REMOTE_DIR:-/scratch/hc676/allo}"
CONDA_BIN="${CONDA_BIN:-/scratch/hc676/allo-agent/bin}"
LLVM_DIR="${LLVM_DIR:-/work/shared/common/llvm-project-main/build}"
VITIS_ENV="${VITIS_ENV:-/work/shared/common/allo/vitis_2023.2_u280.sh}"

TARGET="tests/dataflow/spmw"
if [ $# -gt 0 ]; then
  TARGET="tests/dataflow/spmw/test_spmw_$1.py"
fi

echo "==> checking $HOST is reachable"
if ! ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" true 2>/dev/null; then
  echo "cannot reach $HOST -- is the VPN up?" >&2
  exit 1
fi

echo "==> syncing to $HOST:$REMOTE_DIR"
rsync -az --delete \
  --exclude '.git' --exclude 'build' --exclude 'externals' \
  --exclude '__pycache__' --exclude '*.egg-info' \
  ./ "$HOST:$REMOTE_DIR/"

SETUP="export PATH=$CONDA_BIN:\$PATH; export LLVM_BUILD_DIR=$LLVM_DIR; export OMP_NUM_THREADS=128"
if [ "${HLS:-0}" = "1" ]; then
  SETUP="$SETUP; source $VITIS_ENV"
fi

echo "==> running $TARGET on $HOST"
ssh "$HOST" "bash -lc '$SETUP; cd $REMOTE_DIR && python3 -m pytest $TARGET -v'"
