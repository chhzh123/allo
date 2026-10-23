#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# e1_simtime.sh [reps] [size...]: how long E1's SPMW array takes to simulate,
# functionally and in RTL, REPS times a size (default 2; sizes 4 8 16 32).
# PHASES (default "simulator cosim csim") picks which of the three run:
#   simulator  SPMW's parallel OpenMP simulator: e1_simtime.py, a fresh process
#              a run -> simulator.ndjson
#   cosim      RTL cosimulation with Vivado: spmw_build_array.py --design gemm8
#              --cosim, E1's command, a fresh directory a run. Its log gives the
#              HLS wall time, the array's elaboration and each step of the
#              cosimulation (testbench, IP models, xvlog, xelab, xsim)
#              -> cosim.ndjson and logs/
#   csim       Vitis HLS's own C simulation (csim_design) of the HLS code SPMW
#              emits for the same design: e1_simtime.py <size> csim
#              -> csim.ndjson
# The machine is shared: the load average is kept with every run. T is a
# private copy of this branch; its allo/_mlir links a build of the same C++.
main() {
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
local T=/scratch/hc676/simtime/allo
local E=${SIMTIME_OUT:-/scratch/hc676/e17_simtime}
local REPS=${1:-2}
shift
local SIZES=${*:-4 8 16 32}
mkdir -p $E/logs $E/runs
export PYTHONPATH=$T:$T/tests/dataflow/spmw
local PHASES=${PHASES:-simulator cosim csim}
echo "E1_SIMTIME_START reps=$REPS sizes=$SIZES phases=$PHASES $(date)" >> $E/logs/driver.log
echo "allo: $(python3 -c 'import allo; print(allo.__file__)')" >> $E/logs/driver.log
local S r D t0 t1 rc
case " $PHASES " in *" simulator "*)
for S in $SIZES; do
  for r in $(seq 1 $REPS); do
    python3 $T/experiments/e1_gemm/scripts/e1_simtime.py $S >> $E/simulator.ndjson 2> $E/logs/sim_S${S}_r$r.err
    echo "simulator S=$S r=$r rc=$? $(date +%T)" >> $E/logs/driver.log
  done
done;; esac
case " $PHASES " in *" csim "*)
for S in $SIZES; do
  for r in $(seq 1 $REPS); do
    D=$E/runs/csim_S${S}_r$r
    rm -rf $D
    python3 $T/experiments/e1_gemm/scripts/e1_simtime.py $S csim $D >> $E/csim.ndjson 2> $E/logs/csim_S${S}_r$r.err
    echo "csim S=$S r=$r rc=$? $(date +%T)" >> $E/logs/driver.log
    rm -rf $D/csim_prj/sol/csim/build/*.o
  done
done;; esac
case " $PHASES " in *" cosim "*)
for S in $SIZES; do
  for r in $(seq 1 $REPS); do
    D=$E/runs/cosim_S${S}_r$r
    rm -rf $D
    t0=$(date +%s.%N)
    python3 $T/scripts/spmw_build_array.py --design gemm8 --size $S --frequency 300 \
      --jobs 8 --cosim --out $D > $E/logs/cosim_S${S}_r$r.log 2>&1
    rc=$?
    t1=$(date +%s.%N)
    python3 - "$S" "$r" "$rc" "$t0" "$t1" "$E/logs/cosim_S${S}_r$r.log" "$D/cost.json" >> $E/cosim.ndjson <<'PY'
import json, os, re, sys
size, rep, rc, t0, t1, log, cost = sys.argv[1:]
text = open(log, errors="replace").read()
def first(rx):
    m = re.search(rx, text)
    return float(m.group(1)) if m else None
steps = json.load(open(cost)).get("cosim_seconds", {}) if os.path.isfile(cost) else {}
print(json.dumps({
    "size": int(size), "rep": int(rep), "rc": int(rc),
    "wall_s": round(float(t1) - float(t0), 1),
    "staging_s": first(r"staged \d+ role project\(s\) in ([\d.]+)s"),
    "hls_wall_s": first(r"HLS: ([\d.]+)s wall"),
    "elaborate_s": first(r"Vivado elaborated \d+ instances in ([\d.]+)s"),
    "cosim_steps_s": steps,
    "pass": "SPMW COSIM PASS" in text,
    "cycles": first(r"SPMW CYCLES total=(\d+)"),
    "load": open("/proc/loadavg").read().split()[:3],
}))
PY
    echo "cosim S=$S r=$r rc=$rc $(date +%T)" >> $E/logs/driver.log
    # keep the logs and reports, not the simulation snapshot
    rm -rf $D/sim/xsim.dir $D/ipgen $D/*/prj/sol/.autopilot
  done
done;; esac
echo "E1_SIMTIME_DONE $(date)" >> $E/logs/driver.log
}
main "$@"
