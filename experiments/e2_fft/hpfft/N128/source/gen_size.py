#!/usr/bin/env python3
"""Derive an HP-FFT config for another power-of-two size from a shipped one.

The shipped tree parameterises the size only through FFT_NUM / EXP2_FFT in FFT.h; the stage
pipeline in FFT.cpp is hand-instantiated (one PIPO buffer, one array_partition pragma and one
FFT_stage_spatial_unroll<k> call per stage). This applies exactly the transformation the
authors used between n256 and n1024: append (or remove) trailing stages/buffers and retarget
output_result_array_to_stream to the last buffer. Everything else is copied verbatim.
usage: gen_size.py <src_cfg_dir> <dst_cfg_dir> <N>
"""
import os, re, shutil, sys
src, dst, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
L = N.bit_length() - 1
assert 1 << L == N
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if os.path.isfile(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), dst)
h = open(os.path.join(src, "FFT.h")).read()
Ns = int(re.search(r"^#define FFT_NUM (\d+)", h, re.M).group(1)); Ls = Ns.bit_length() - 1
h = re.sub(r"^#define FFT_NUM \d+", f"#define FFT_NUM {N}", h, flags=re.M)
h = re.sub(r"^#define EXP2_FFT \d+", f"#define EXP2_FFT {L}", h, flags=re.M)
open(os.path.join(dst, "FFT.h"), "w").write(h)
c = open(os.path.join(src, "FFT.cpp")).read()
calls = re.findall(r"^([ \t]*)FFT_stage_spatial_unroll<(\d+)>\(data_(\d+), data_(\d+)\);[ \t]*\n", c, re.M)
if not calls:
    # array-interface baselines (no_StagePipeline, original_C_style) loop over EXP2_FFT: header change is enough
    open(os.path.join(dst, "FFT.cpp"), "w").write(c); print("header-only", src, "->", dst); sys.exit(0)
last_stage = max(int(x[1]) for x in calls); assert last_stage == Ls, (last_stage, Ls)
last_data = max(int(x[3]) for x in calls)
d = L - Ls
if d > 0:
    indent = calls[-1][0]
    add = "".join(f"{indent}FFT_stage_spatial_unroll<{last_stage+i}>(data_{last_data+i-1}, data_{last_data+i});\n" for i in range(1, d+1))
    c, n = re.subn(rf"(^[ \t]*FFT_stage_spatial_unroll<{last_stage}>\(data_{last_data-1}, data_{last_data}\);[ \t]*\n)", lambda m: m.group(1) + add, c, flags=re.M); assert n == 1
    m = re.search(rf"^([ \t]*)static complex<float> data_{last_data}\[FFT_NUM\];[ \t]*\n", c, re.M); assert m
    c = c[:m.end()] + "".join(f"{m.group(1)}static complex<float> data_{last_data+i}[FFT_NUM];\n" for i in range(1, d+1)) + c[m.end():]
    m = re.search(rf"^([ \t]*)#pragma HLS array_partition variable=data_{last_data} type=cyclic factor=UF dim=1[ \t]*\n", c, re.M); assert m
    c = c[:m.end()] + "".join(f"{m.group(1)}#pragma HLS array_partition variable=data_{last_data+i} type=cyclic factor=UF dim=1\n" for i in range(1, d+1)) + c[m.end():]
elif d < 0:
    for i in range(-d):
        s_, dd = last_stage - i, last_data - i
        c, n1 = re.subn(rf"^[ \t]*FFT_stage_spatial_unroll<{s_}>\(data_{dd-1}, data_{dd}\);[ \t]*\n", "", c, flags=re.M)
        c, n2 = re.subn(rf"^[ \t]*static complex<float> data_{dd}\[FFT_NUM\];[ \t]*\n", "", c, flags=re.M)
        c, n3 = re.subn(rf"^[ \t]*#pragma HLS array_partition variable=data_{dd} type=cyclic factor=UF dim=1[ \t]*\n", "", c, flags=re.M)
        assert (n1, n2, n3) == (1, 1, 1), (n1, n2, n3, s_, dd)
c, n = re.subn(rf"output_result_array_to_stream \(data_{last_data}, dataOut\)", f"output_result_array_to_stream (data_{last_data+d}, dataOut)", c); assert n == 1
open(os.path.join(dst, "FFT.cpp"), "w").write(c)
print(f"generated {dst}: N={N} stages 2..{L} from {src} (N={Ns})")
