#!/bin/bash
# probe_hls.sh <size>: stage one ptpu build's roles and csynth one of each kind.
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
export PYTHONPATH=/scratch/hc676/allo:/scratch/hc676/allo/tests/dataflow/spmw
export TMPDIR=/scratch/hc676/e12_ptpu/tmp
S=$1
OUT=/scratch/hc676/e12_ptpu/probe_$S
rm -rf "$OUT"; mkdir -p "$OUT"
cd /scratch/hc676/allo
python3 - <<PY
import sys
sys.argv = ["x"]
sys.path.insert(0, "scripts")
import spmw_build_array as B
import allo.spmw as spmw
fabric = B.design("ptpu-mixed", $S)
graph = spmw.elaborate(fabric)
names = B.stage(graph, "$OUT", B.PART, 300.0)
print("ROLES", names)
PY
cd "$OUT"
for r in $(ls -d *_r* | sort); do
  ( cd $r && sed -i '/export_design/d' run.tcl && vitis_hls -f run.tcl > vitis_hls.log 2>&1 ) &
done
wait
for r in $(ls -d *_r* | sort); do
  rpt=$r/prj/sol/syn/report/${r}_0_csynth.rpt
  clk=$(grep -m1 "|ap_clk" $rpt | tr -s " ")
  loop=$(grep -E "^\s*\|[-+ ]+[A-Za-z_]" $rpt | grep -i yes | head -1 | tr -s " ")
  tot=$(grep -m1 "^|Total " $rpt | tr -s " ")
  sch=$r/prj/sol/.autopilot/db/${r}_0.verbose.sched.rpt
  acc=$(grep -oE "^ST_[0-9]+ : Operation [0-9]+ \[1/1\] \([0-9.]+ns\)   --->   \"%[A-Za-z0-9_]+ = (read|write) [a-z0-9]+ @_ssdm_op_(Read|Write)\.ap_fifo[^,]*, i[0-9]+ %v[0-9]+" $sch | sed -E 's/^(ST_[0-9]+).* = (read|write) .*%(v[0-9]+)$/\1:\2:\3/' | tr "\n" " ")
  echo "== $r  $clk  $loop  $tot"
  echo "     $acc"
done
