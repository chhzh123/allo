#!/bin/bash
# Build the kernel with the array's floorplan actually in effect.
#
# The floorplan has to be injected into the platform's own `preopt.tcl` hook,
# which is the only place it takes: a scoped XDC in the IP creates the pblocks
# but loses their cells across the RM->top flattening, and a TCL.PRE property
# is overwritten by vpl, which claims every hook it uses.
#
# `preopt.tcl` is copied fresh from the platform each time v++ sets up, so the
# injection cannot be done beforehand. It has to land after setup and before
# opt_design -- a window synthesis leaves wide open, about half an hour.
set -e
OUT=$1
HOOK=$OUT/vpp/link/vivado/vpl/.local/hw_platform/tcl_hooks/preopt.tcl

cd /scratch/hc676/allo
export PATH=/scratch/hc676/allo-agent/bin:$PATH
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
ulimit -n 8192

rm -rf "$OUT"
nohup python3 scripts/spmw_package_kernel.py --design transformer16 --size 16 \
  --out "$OUT" --slots 4 --frequency 300 --link-frequency 250 --jobs 8 \
  > "$OUT.log" 2>&1 &
BUILD=$!
echo "build pid $BUILD"

# Wait for the platform hook to be laid down, then inject before opt_design.
for i in $(seq 1 240); do
  [ -f "$HOOK" ] && break
  sleep 30
done
[ -f "$HOOK" ] || { echo "INJECT FAILED: hook never appeared"; exit 1; }
sleep 60   # let v++ finish copying .local before touching it

PYTHONPATH=tests/dataflow/spmw:scripts python3 - "$OUT" <<'PY'
import sys
import allo.spmw as spmw
from allo.spmw.shell import floorplan_xdc
from spmw_build_array import design
graph = spmw.elaborate(design("transformer16", 16))
open(sys.argv[1] + "/floorplan.tcl", "w").write(
    floorplan_xdc(graph, top="level0_i/ulp/spmw_kernel_1/inst/dut", slots=4,
                  parent="pblock_dynamic_region", cols=6))
PY
{
  echo ""
  echo "# ---- SPMW FLOORPLAN ----"
  cat "$OUT/floorplan.tcl"
  echo "# ---- end SPMW FLOORPLAN ----"
} >> "$HOOK"
echo "INJECTED $(grep -c create_pblock "$OUT/floorplan.tcl") pblocks at $(date)"
grep -c "SPMW FLOORPLAN" "$HOOK"
wait $BUILD
echo "BUILD DONE rc=$?"
