#!/bin/bash
# Build and grade one SystemVerilog trial. usage: build.sh <trialdir> <vecset>
# The agent writes design.sv containing module dut_norm; nothing else is added.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
T=$1; V=${2:-visible}; S=/scratch/hc676/agentstudy
cd "$T" || exit 2
rm -rf sim; mkdir -p sim; cd sim
cp "$T/design.sv" . 2>/dev/null || { echo "STUDY BUILD no design.sv"; exit 2; }
cp "$S/task/tb_study.sv" .
NPROD=$(python3 -c "import json;print(len(json.load(open('$S/task/vectors/$V.json'))))")
T0=$(date +%s.%N); xvlog -sv design.sv tb_study.sv > xvlog.log 2>&1 || { echo "STUDY BUILD FAIL compile"; grep -iE "^ERROR|error:" xvlog.log | head -20; exit 1; }
xelab tb -s tbsim --generic_top "NPROD=$NPROD" -L unisims_ver -L unimacro_ver -L secureip > xelab.log 2>&1 || { echo "STUDY BUILD FAIL elaborate"; grep -iE "^ERROR" xelab.log | head -20; exit 1; }
echo "STUDY STAGE elaborate $(echo \"$(date +%s.%N) - $T0\" | bc)"; T0=$(date +%s.%N)
xsim tbsim -runall -testplusarg "vecdir=$S/task/vectors/$V" > xsim.log 2>&1
echo "STUDY STAGE simulate $(echo \"$(date +%s.%N) - $T0\" | bc)"
grep -E "STUDY |MISMATCH" xsim.log | head -30
grep -q "STUDY RESULT" xsim.log || { echo "STUDY BUILD FAIL simulation produced no verdict"; tail -5 xsim.log; exit 1; }
