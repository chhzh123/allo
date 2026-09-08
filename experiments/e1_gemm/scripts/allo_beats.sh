#!/bin/bash
# Measure Allo on the same definition as SPMW and AutoSA: first memory beat to
# last, one launch. The shipped Allo cosim ran three seeds per launch, which
# would smear the window across three launches, so the measurement build runs
# one seed. Correctness is unchanged -- the three-seed run is the shipped one.
set -u
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
S=$1
R=/scratch/hc676/spmw_eval_remaining_2026-09-06/e1_gemm/allo/reports/allo_S${S}_tile_cosim
W=/scratch/hc676/e1_allo_beats/S$S
[ -f $R/kernel.cpp ] || { echo "S=$S no shipped kernel"; exit 1; }
rm -rf $W; mkdir -p $W; cd $W
cp $R/kernel.cpp $R/tb.cpp .
sed -i "s/const int seeds\[\] = {0, 1, 2};/const int seeds[] = {0};/" tb.cpp
grep -q "seeds\[\] = {0};" tb.cpp || { echo "S=$S could not reduce to one seed"; exit 1; }
cat > kernel.h <<'H'
#ifndef KERNEL_H
#define KERNEL_H
#include <stdint.h>
void gemm(int8_t *A, int8_t *B, int32_t *C);
#endif
H
# Inputs the testbench accepts: it rejects anything not spanning the int8
# extrema, so plant them explicitly.
/scratch/hc676/allo-agent/bin/python3 - "$S" <<'PY'
import numpy as np, sys
S = int(sys.argv[1])
rng = np.random.default_rng(0)
for nm, shape in (("A0", (S, S)), ("B0", (S, S))):
    x = rng.integers(-128, 128, size=shape, dtype=np.int64).astype(np.int8)
    flat = x.reshape(-1); flat[0] = -128; flat[1] = 127     # span the extrema
    x.astype(np.int8).tofile(nm + ".bin")
A = np.fromfile("A0.bin", dtype=np.int8).reshape(S, S).astype(np.int32)
B = np.fromfile("B0.bin", dtype=np.int8).reshape(S, S).astype(np.int32)
(A @ B).astype(np.int32).tofile("Cref0.bin")
PY
cat > run.tcl <<TCL
open_project prj -reset
open_solution -reset sol -flow_target vivado
set_top gemm
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=gnu++0x"
add_files -tb A0.bin
add_files -tb B0.bin
add_files -tb Cref0.bin
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.333
csynth_design
cosim_design -rtl verilog
exit
TCL
T0=$(date +%s)
vitis_hls -f run.tcl > build.log 2>&1
echo "S=$S rc=$? wall_s=$(( $(date +%s) - T0 ))"
grep -E "E1 TB|Passed!|co-simulation finished" build.log | head -4
grep -E "^\| *Verilog\|" prj/sol/sim/report/gemm_cosim.rpt 2>/dev/null | cut -c1-110
echo "ALLO_BEATS_HLS_DONE_$S"
