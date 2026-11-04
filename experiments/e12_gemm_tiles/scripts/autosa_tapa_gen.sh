#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# autosa_tapa_gen.sh <W> <KTILE>: E12's AutoSA row through --target=autosa_tapa. Same input
# (E12's kernel.c/kernel.h for W^3) and options as E12's autosa_tiles2.sh build:
# 16x16 cells, --host-serialize, latency[4,4], array_part[64,64,KTILE], simd[1].
# Then E1's gen.sh changes for TAPA 0.1.20250127 (tapa::mmap kernel signatures,
# kernel0 prototype into the host, tapa::invoke + gflags), the host build and
# TAPA's software simulation against AutoSA's own golden model.
main() {
set -u
local W=$1 KT=$2 S=16 L=4 E=/scratch/hc676/e22_autosa_tapa/e12
local SRC=/scratch/hc676/e16_tiles/autosa/wide_k${KT}_l$L/S${S}_W$W
local D=$E/gen/W$W
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
rm -rf $D; mkdir -p $D/out/src $D/out/latency_est $D/out/resource_est $D/out/tuning
cp $SRC/kernel.c $SRC/kernel.h $SRC/simd_info.json $D/
local SA="{kernel[]->space_time[3];kernel[]->array_part[$((S*L)),$((S*L)),$KT];kernel[]->latency[$L,$L];kernel[]->simd[1]}"
echo "$SA" > $D/sa_sizes.txt
(export PATH=/scratch/hc676/allo-agent/bin:/scratch/hc676/autosa/llvm9/bin:$PATH
 export LD_LIBRARY_PATH=/scratch/hc676/autosa/llvm9/lib:${LD_LIBRARY_PATH:-}
 cd /scratch/hc676/autosa/AutoSA && ./autosa $D/kernel.c --config=./autosa_config/autosa_config.json \
   --target=autosa_tapa --output-dir=$D/out --host-serialize --sa-sizes="$SA" \
   --simd-info=$D/simd_info.json --hls > $D/gen.log 2>&1) || { echo "W=$W autosa failed"; return 1; }
local G=$D/out/src
sed -i -E 's/tapa::(read_only|write_only)_mmap</tapa::mmap</g' $G/kernel_kernel.cpp $G/kernel_kernel.h
python3 - $G <<'PY'
import re, sys
d = sys.argv[1]
h = open(d + "/kernel_kernel.h").read()
proto = re.search(r"^void kernel0\(.*?\);\n", h, re.M | re.S).group(0)
open(d + "/kernel_kernel.h", "w").write(h.replace(proto, ""))
c = open(d + "/kernel_host.cpp").read()
inc = '#include "kernel_kernel.h"\n'
assert inc in c
open(d + "/kernel_host.cpp", "w").write(c.replace(inc, inc + proto, 1))
PY
sed -i 's|tapa::task().invoke(kernel0, |tapa::invoke(kernel0, getenv("TAPA_BITSTREAM") ? getenv("TAPA_BITSTREAM") : "", |' $G/kernel_host.cpp
sed -i 's|^int main(int argc, char \*\*argv) {|#include <gflags/gflags.h>\nint main(int argc, char **argv) {\n  gflags::ParseCommandLineFlags(\&argc, \&argv, true);|' $G/kernel_host.cpp
sed -i -E 's/(buffer_[A-Z]+\[0\])\.vectorized<1>\(\)/std::move(\1)/g' $G/kernel_host.cpp
(cd $G && tapa g++ -- kernel_host.cpp kernel_kernel.cpp -O2 -o host > gxx.log 2>&1) || { echo "W=$W host build failed"; return 1; }
local t0=$(date +%s)
(cd $G && timeout 3600 ./host > swsim.log 2>&1)
echo "W=$W KT=$KT software sim: $(grep -o 'Passed!\|Failed with [0-9]* errors' $G/swsim.log | head -1) ($(( $(date +%s) - t0 )) s) | tasks $(grep -oE '\.invoke\(([A-Za-z0-9_]+)' $G/kernel_kernel.cpp | sort -u | wc -l), instances $(grep -c '\.invoke' $G/kernel_kernel.cpp), PE instances $(grep -c 'invoke(PE_wrapper' $G/kernel_kernel.cpp)"
}
main "$@"
