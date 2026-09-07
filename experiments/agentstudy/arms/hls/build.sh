#!/bin/bash
# Build and grade one Vitis HLS trial. usage: build.sh <trialdir> <vecset>
# The agent writes design.cpp defining void gemm_tile(...); this script
# synthesises it, wraps the exported RTL to the study's port names, and runs
# the same testbench every arm runs.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
T=$1; V=${2:-visible}; S=/scratch/hc676/agentstudy
cd "$T" || exit 2
[ -f design.cpp ] || { echo "STUDY BUILD no design.cpp"; exit 2; }
rm -rf prj sim
cat > run.tcl <<'TCL'
open_project prj
set_top gemm_tile
add_files design.cpp
open_solution sol
set_part xcu280-fsvh2892-2L-e
create_clock -period 3.33 -name default
config_interface -clock_enable=0
config_rtl -reset_level low
config_op mul -impl dsp
csynth_design
export_design -format ip_catalog
exit
TCL
T0=$(date +%s.%N)
vitis_hls -f run.tcl > hls.log 2>&1
echo "STUDY STAGE synthesise $(echo \"$(date +%s.%N) - $T0\" | bc)"
V_DIR=prj/sol/syn/verilog
[ -f "$V_DIR/gemm_tile.v" ] || { echo "STUDY BUILD FAIL synthesis"; grep -iE "^ERROR|error:" hls.log | head -20; exit 1; }
python3 "$S/arms/hls/wrap.py" "$V_DIR/gemm_tile.v" > dut_norm.sv || { echo "STUDY BUILD FAIL wrapper"; exit 1; }
grep -E "^\| *\+ *gemm_tile" -A0 prj/sol/syn/report/csynth.rpt 2>/dev/null | head -2
mkdir -p sim; cd sim
cp ../dut_norm.sv "$S/task/tb_study.sv" .
cp ../$V_DIR/*.v . 2>/dev/null
NPROD=$(python3 -c "import json;print(len(json.load(open('$S/task/vectors/$V.json'))))")
T0=$(date +%s.%N); xvlog -sv dut_norm.sv tb_study.sv > xvlog.log 2>&1 || { echo "STUDY BUILD FAIL compile"; grep -iE "^ERROR" xvlog.log | head; exit 1; }
xvlog *.v >> xvlog.log 2>&1
echo "STUDY STAGE compile $(echo \"$(date +%s.%N) - $T0\" | bc)"; T0=$(date +%s.%N)
xelab tb -s tbsim --timescale 1ns/1ps --generic_top "NPROD=$NPROD" -L unisims_ver -L unimacro_ver -L secureip > xelab.log 2>&1 || { echo "STUDY BUILD FAIL elaborate"; grep -iE "^ERROR" xelab.log | head; exit 1; }
echo "STUDY STAGE elaborate $(echo \"$(date +%s.%N) - $T0\" | bc)"; T0=$(date +%s.%N)
xsim tbsim -runall -testplusarg "vecdir=$S/task/vectors/$V" > xsim.log 2>&1
echo "STUDY STAGE simulate $(echo \"$(date +%s.%N) - $T0\" | bc)"
grep -E "STUDY |MISMATCH" xsim.log | head -30
grep -q "STUDY RESULT" xsim.log || { echo "STUDY BUILD FAIL simulation produced no verdict"; tail -5 xsim.log; exit 1; }
