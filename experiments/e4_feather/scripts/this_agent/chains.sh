#!/bin/bash
# E4b run chains: general weights (--pattern full: every stored byte in [0, 256),
# the same bytes on both sides for seed 0; the SPMW port reads them as int8,
# the RTL as uint8 with zero point 0), whole workloads, both systems.
#   chains.sh rtlA | rtlB | rtlC | spmw
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build PYTHONDONTWRITEBYTECODE=1
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
W=/scratch/hc676/e4b_work
R=/scratch/hc676/e4b_rtl_runs
S=/scratch/hc676/e4b_spmw
P=/scratch/hc676/e4_spmw
FIXED=/scratch/hc676/e4_feather_fixed/RTL
ORIG=/scratch/hc676/feather_ref/FEATHER_RTL/RTL
cd $W

rtl() {  # rtl <run name> <RTL dir> <N> <generator args...>
  local name=$1 rtl=$2 N=$3; shift 3
  if [ -f $R/$name/check.json ]; then echo "skip $name"; return; fi
  echo "== $name $(date)"
  T0=$(date +%s)
  python3 -u e4_rtl_run.py --rtl $rtl --N $N --out $R/$name "$@" > $R/$name.log 2>&1
  echo "$name rc=$? wall=$(( $(date +%s) - T0 ))s: $(tail -n 1 $R/$name.log | cut -c1-400)"
}

spmw() {  # spmw <run name> <driver args...>
  local name=$1; shift
  if [ -f $S/$name/result.json ]; then echo "skip $name"; return; fi
  echo "== $name $(date)"
  T0=$(date +%s)
  python3 -u e4_spmw_run.py --out $S/$name "$@" > $S/$name.log 2>&1
  echo "$name rc=$? wall=$(( $(date +%s) - T0 ))s: $(tail -n 1 $S/$name.log | cut -c1-400)"
}

reuse() {  # copy a finished build (HLS'd roles, assembled array, IP sim models) without its run
  [ -f $S/$1/roles.json ] || rsync -a --exclude sim --exclude result.json --exclude operands.npz $P/$1/ $S/$1/
}

GEMM="--workload gemm --gemm 128,128,128 --pattern full --seed 0"
CONV="--workload conv --conv 64,16,16,64 --pattern full --seed 0"

case $1 in
rtlA)
  rtl wl_gemm_N4_m0 $FIXED 4 $GEMM --mode 0 --zpa 0 --zpw 0
  rtl wl_gemm_N4_m1 $FIXED 4 $GEMM --mode 1 --zpa 0 --zpw 0
  rtl wl_gemm_N8_m0 $FIXED 8 $GEMM --mode 0 --zpa 0 --zpw 0
  rtl wl_gemm_N8_m1 $FIXED 8 $GEMM --mode 1 --zpa 0 --zpw 0
  rtl wl_conv_N4_m0 $FIXED 4 $CONV --mode 0 --zpa 0 --zpw 0
  rtl wl_conv_N4_m1 $FIXED 4 $CONV --mode 1 --zpa 0 --zpw 0
  # operands on both sides of the zero point: what the RTL's unsigned datapath makes of them
  for N in 4 8 16; do
    rtl demo_mixed_N${N} $FIXED $N --workload tile --tiles 4 --program gemm --pattern mixed --zpa 128 --zpw 128 --seed 0 --mode 1
  done
  # the shipped controller on a general-weight workload
  rtl wl_gemm_N4_m0_orig $ORIG 4 $GEMM --mode 0 --zpa 0 --zpw 0
  ;;
rtlB)
  rtl wl_conv_N8_m0 $FIXED 8 $CONV --mode 0 --zpa 0 --zpw 0
  rtl wl_conv_N8_m1 $FIXED 8 $CONV --mode 1 --zpa 0 --zpw 0
  rtl wl_gemm_N16_m0 $FIXED 16 $GEMM --mode 0 --zpa 0 --zpw 0
  rtl wl_gemm_N16_m1 $FIXED 16 $GEMM --mode 1 --zpa 0 --zpw 0
  ;;
rtlC)
  rtl wl_conv_N16_m1 $FIXED 16 $CONV --mode 1 --zpa 0 --zpw 0
  rtl wl_conv_N16_m0 $FIXED 16 $CONV --mode 0 --zpa 0 --zpw 0
  ;;
spmw)
  for d in wl_gemm_N8_stream wl_gemm_N8_x wl_gemm_N16_stream wl_gemm_N16_x wl_conv_N4_stream wl_conv_N4_x; do reuse $d; done
  spmw wl_gemm_N8_stream  --N 8  $GEMM --reuse-build
  spmw wl_gemm_N8_x       --N 8  $GEMM --reuse-build --resident
  spmw wl_conv_N4_stream  --N 4  $CONV --reuse-build
  spmw wl_conv_N4_x       --N 4  $CONV --reuse-build --resident
  spmw wl_gemm_N16_stream --N 16 $GEMM --reuse-build
  spmw wl_gemm_N16_x      --N 16 $GEMM --reuse-build --resident
  spmw wl_gemm_N4_stream  --N 4  $GEMM --jobs 4
  spmw wl_gemm_N4_x       --N 4  $GEMM --jobs 4 --resident
  spmw wl_conv_N8_stream  --N 8  $CONV --jobs 4
  spmw wl_conv_N8_x       --N 8  $CONV --jobs 4 --resident
  spmw wl_conv_N16_stream --N 16 $CONV --jobs 4
  spmw wl_conv_N16_x      --N 16 $CONV --jobs 4 --resident
  ;;
esac
echo "CHAIN $1 DONE $(date)"
