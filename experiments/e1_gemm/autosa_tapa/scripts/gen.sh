#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# gen.sh <S> [narrow]: AutoSA with --target=autosa_tapa, the E1 options (array_part[S,S,S],
# latency[1,1], simd[1], --host-serialize unless "narrow").
# The 2021 TAPA backend writes tapa::read_only_mmap / write_only_mmap in kernel
# signatures; TAPA 0.1.20250127 kernels take tapa::mmap (the host side keeps the
# directional types), so the kernel files get that rename. TAPA's rewrite of the
# top task's prototype also swallows the next declaration's first token when the
# prototype sits in the shared header, so the prototype moves to the host file,
# where TAPA's own examples declare the kernel.
main() {
set -u
export PATH=/scratch/hc676/allo-agent/bin:/scratch/hc676/autosa/llvm9/bin:$PATH
export LD_LIBRARY_PATH=/scratch/hc676/autosa/llvm9/lib:${LD_LIBRARY_PATH:-}
local S=$1 MODE=${2:-wide} W=/scratch/hc676/e22_autosa_tapa
local D=$W/gen/S$S EXTRA=--host-serialize
[ "$MODE" = narrow ] && EXTRA=""
rm -rf $D; mkdir -p $D/out/src $D/out/latency_est $D/out/resource_est $D/out/tuning
cp /scratch/hc676/e1_autosa_wide/S8/kernel.c /scratch/hc676/e1_autosa_wide/S8/simd_info.json $D/
sed -E "s/#define I [0-9]+/#define I $S/; s/#define J [0-9]+/#define J $S/; s/#define K [0-9]+/#define K $S/" \
  /scratch/hc676/e1_autosa_wide/S8/kernel.h > $D/kernel.h
cd /scratch/hc676/autosa/AutoSA
./autosa $D/kernel.c --config=./autosa_config/autosa_config.json --target=autosa_tapa \
  --output-dir=$D/out $EXTRA \
  --sa-sizes="{kernel[]->space_time[3];kernel[]->array_part[$S,$S,$S];kernel[]->latency[1,1];kernel[]->simd[1]}" \
  --simd-info=$D/simd_info.json --hls > $D/gen.log 2>&1 || { echo "S=$S autosa failed"; return 1; }
sed -i -E 's/tapa::(read_only|write_only)_mmap</tapa::mmap</g' $D/out/src/kernel_kernel.cpp $D/out/src/kernel_kernel.h
python3 - $D/out/src <<'PY'
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
# Host: tapa::task().invoke only runs the software model; tapa::invoke with a
# bitstream path (TAPA_BITSTREAM, empty = software) also runs RTL cosim, and
# gflags parsing exposes TAPA's --xosim_* options.
sed -i 's|tapa::task().invoke(kernel0, |tapa::invoke(kernel0, getenv("TAPA_BITSTREAM") ? getenv("TAPA_BITSTREAM") : "", |' $D/out/src/kernel_host.cpp
sed -i 's|^int main(int argc, char \*\*argv) {|#include <gflags/gflags.h>\nint main(int argc, char **argv) {\n  gflags::ParseCommandLineFlags(\&argc, \&argv, true);|' $D/out/src/kernel_host.cpp
# A scalar port (narrow C) is passed as .vectorized<1>(), which tapa::mmap<int> rejects;
# pass the mmap itself, as a temporary (an lvalue is taken for a scalar argument).
sed -i -E 's/(buffer_[A-Z]+\[0\])\.vectorized<1>\(\)/std::move(\1)/g' $D/out/src/kernel_host.cpp
source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
(cd $D/out/src && tapa g++ -- kernel_host.cpp kernel_kernel.cpp -O2 -o host > gxx.log 2>&1) || { echo "S=$S host build failed"; return 1; }
(cd $D/out/src && ./host > swsim.log 2>&1)
echo "S=$S $MODE software sim: $(grep -o 'Passed!\|Failed with [0-9]* errors' $D/out/src/swsim.log)"
echo "S=$S $MODE generated: $(grep -c '\.invoke' $D/out/src/kernel_kernel.cpp) invokes"
}
main "$@"
