#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# e9_simtime.sh [reps] [program...]: how long E9's programmable mini-TPU
# takes to simulate its long programs at 4x4 (SIZE), functionally and in RTL,
# REPS times each (default 2; programs llama dsv4). PHASES (default
# "simulator csim cosim") picks which of the three run:
#   simulator  SPMW's parallel OpenMP simulator: e9_simtime.py, a fresh process
#              a run -> simulator.ndjson
#   csim       Vitis HLS's own C simulation of the HLS code SPMW emits, its
#              calls put producer first: e9_simtime.py <program> <size> csim
#              -> csim.ndjson
#   cosim      RTL cosimulation with Vivado: spmw_build_array.py --design
#              ptpu-<program> --cosim, E9's command, a fresh directory a run;
#              its log gives the HLS wall time, the array's elaboration and
#              each cosimulation step -> cosim.ndjson and logs/
# The machine is shared: the load average is kept with every run. T is a
# private copy of this branch; its allo/_mlir links a build of the same C++.
main() {
set -u
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
local T=/scratch/hc676/simtime/allo
local E=/scratch/hc676/e17_simtime/e9
local SIZE=${SIZE:-4}
# The generated top copies each operand into a local array: megabytes for
# these programs, so Vitis's C simulation needs the stack unbounded.
ulimit -s unlimited
local REPS=${1:-2}
shift
local PROGRAMS=${*:-llama dsv4}
mkdir -p $E/logs $E/runs
export PYTHONPATH=$T:$T/tests/dataflow/spmw
local PHASES=${PHASES:-simulator csim cosim}
echo "E9_SIMTIME_START reps=$REPS size=$SIZE programs=$PROGRAMS phases=$PHASES $(date)" >> $E/logs/driver.log
echo "allo: $(python3 -c 'import allo; print(allo.__file__)')" >> $E/logs/driver.log
local P r D t0 t1 rc
case " $PHASES " in *" simulator "*)
for P in $PROGRAMS; do
  for r in $(seq 1 $REPS); do
    python3 $T/experiments/e9_vta/scripts/e9_simtime.py $P $SIZE >> $E/simulator.ndjson 2> $E/logs/sim_S${SIZE}_${P}_r$r.err
    echo "simulator S=$SIZE $P r=$r rc=$? $(date +%T)" >> $E/logs/driver.log
  done
done;; esac
case " $PHASES " in *" csim "*)
for P in $PROGRAMS; do
  for r in $(seq 1 $REPS); do
    D=$E/runs/csim_S${SIZE}_${P}_r$r
    rm -rf $D
    python3 $T/experiments/e9_vta/scripts/e9_simtime.py $P $SIZE csim $D >> $E/csim.ndjson 2> $E/logs/csim_S${SIZE}_${P}_r$r.err
    echo "csim S=$SIZE $P r=$r rc=$? $(date +%T)" >> $E/logs/driver.log
    rm -rf $D/csim_prj/sol/csim/build/*.o
  done
done;; esac
case " $PHASES " in *" cosim "*)
for P in $PROGRAMS; do
  for r in $(seq 1 $REPS); do
    D=$E/runs/cosim_S${SIZE}_${P}_r$r
    rm -rf $D
    t0=$(date +%s.%N)
    python3 $T/scripts/spmw_build_array.py --design ptpu-$P --size $SIZE --frequency 300 \
      --jobs 6 --cosim --out $D > $E/logs/cosim_S${SIZE}_${P}_r$r.log 2>&1
    rc=$?
    t1=$(date +%s.%N)
    python3 - "$P" "$r" "$rc" "$t0" "$t1" "$E/logs/cosim_S${SIZE}_${P}_r$r.log" "$D/cost.json" "$SIZE" >> $E/cosim.ndjson <<'PY'
import json, os, re, sys
program, rep, rc, t0, t1, log, cost, size = sys.argv[1:]
text = open(log, errors="replace").read()
def first(rx):
    m = re.search(rx, text)
    return float(m.group(1)) if m else None
steps = json.load(open(cost)).get("cosim_seconds", {}) if os.path.isfile(cost) else {}
print(json.dumps({
    "program": program, "size": int(size), "rep": int(rep), "rc": int(rc),
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
    echo "cosim S=$SIZE $P r=$r rc=$rc $(date +%T)" >> $E/logs/driver.log
    # keep the logs and reports, not the simulation snapshot
    rm -rf $D/sim/xsim.dir $D/ipgen $D/*/prj/sol/.autopilot
  done
done;; esac
echo "E9_SIMTIME_DONE $(date)" >> $E/logs/driver.log
}
main "$@"
