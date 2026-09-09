#!/usr/bin/env python3
"""E1 baseline: generate one Allo library-systolic GEMM build directory.

usage: e1_allo_gen.py S WORKLOAD OUTDIR
  S         array size (Mt = Nt = S)
  WORKLOAD  'tile' (M = N = K = S, one tile) or an integer (M = N = K = that)

What it does, in order:
  1. wraps allo.library.systolic in a top named `gemm` (Allo's vitis_hls
     post-processing keys on `line.startswith("void " + top)`, so a top named
     `systolic` also mangles `systolic_tile`; the wrapper sidesteps that);
  2. composes the library's own schedule (`schedule_systolic`, applied by
     `Schedule.compose` through KERNEL2SCHEDULE) and lets Allo emit the Vitis
     kernel (kernel.cpp with m_axi interfaces, kernel.h) into OUTDIR;
  3. checks the emitted PE is output-stationary (one C store per PE, outside
     the k loop) and that the tile loop folds temporally;
  4. writes the int8 inputs for seeds 0, 1, 2 (mixed sign, extrema forced),
     the int32 numpy reference, a C testbench that fails on any mismatch, and
     run_e1.tcl: Allo's codegen_tcl template with csim, csynth, cosim and
     export -flow impl, every stage timed.
Nothing here runs a tool; e1_allo_run.sh does that.
"""
import json
import os
import re
import sys

import numpy as np

import allo
from allo.ir.types import int8, int32
from allo.library.systolic import systolic

PART = "xcu280-fsvh2892-2L-e"
PERIOD_NS = "3.333"  # 300 MHz, the eval's period; Allo's own template rounds to 3.33
SEEDS = (0, 1, 2)


def build_schedule(M, K, N, S):
    def gemm(A: int8[M, K], B: int8[K, N], C: int32[M, N]):
        systolic[int8, int8, int32, M, K, N, S, S, "gemm"](A, B, C)

    s = allo.customize(gemm)
    # KERNEL2SCHEDULE[systolic] is schedule_systolic: partitions, pipelines,
    # fuses the tile loops, unfolds the PE grid and streams A/B between PEs.
    s.compose(systolic, instantiate=[int8, int8, int32, M, K, N, S, S], id="gemm")
    return s


def pe_functions(code):
    """Names of the emitted PE functions (one per spatial PE after unfold)."""
    return re.findall(r"^void (PE_kernel\w*)\(", code, flags=re.M)


def function_body(code, name):
    start = code.index(f"void {name}(")
    depth = 0
    i = code.index("{", start)
    for j in range(i, len(code)):
        if code[j] == "{":
            depth += 1
        elif code[j] == "}":
            depth -= 1
            if depth == 0:
                return code[start:j + 1]
    raise ValueError(name)


def check_output_stationary(code):
    """The PE is output-stationary iff it stores to its C argument exactly once,
    at brace depth 1 (function level, after the k loop), and the k loop body
    carries the accumulation. Returns (ok, detail)."""
    names = pe_functions(code)
    if not names:
        return False, {"error": "no PE_kernel function emitted"}
    body = function_body(code, names[0])
    header = body[: body.index("{")]
    args = [a.strip() for a in header[header.index("(") + 1: header.rindex(")")].split(",")]
    c_arg = None
    for a in args:
        m = re.match(r"int32_t (\w+)\[", a)
        if m:
            c_arg = m.group(1)
    if c_arg is None:
        return False, {"error": f"no int32 array argument in {names[0]}", "args": args}
    depth = 0
    stores = []
    loop_depth_stores = 0
    for ln in body.splitlines():
        stripped = ln.strip()
        if re.match(rf"{c_arg}\[.*\]\[.*\] = ", stripped):
            stores.append((depth, stripped))
            if depth > 1:
                loop_depth_stores += 1
        depth += ln.count("{") - ln.count("}")
    acc = re.search(r"\bfor \(int (\w+) = 0; \1 < (\d+); \1\+\+\)", body)
    ok = len(stores) == 1 and stores[0][0] == 1 and loop_depth_stores == 0 and acc is not None
    detail = {
        "pe_function": names[0],
        "pe_count": len(names),
        "c_stores": [s for _, s in stores],
        "c_store_depths": [d for d, _ in stores],
        "k_loop_trip": int(acc.group(2)) if acc else None,
        "pipeline_pragma": "#pragma HLS pipeline II=1" in body,
    }
    return ok, detail


def tile_loop_info(code, M, N, S):
    """The library's fused outer_tile loop in systolic_gemm: its trip count
    must be (M/S)*(N/S) -- tiles are folded temporally, one after another."""
    expected = (M // S) * (N // S)
    try:
        body = function_body(code, "systolic_gemm")
    except ValueError:
        return {"error": "no systolic_gemm function emitted", "expected": expected}
    loops = re.findall(r"^(\s*)(\w*): for \(int \w+ = 0; \w+ < (\d+); \w+\+\+\)", body, flags=re.M)
    if not loops:
        return {"error": "no loops found in systolic_gemm", "expected": expected}
    outer = min(loops, key=lambda t: len(t[0]))
    return {"outer_loop": outer[1], "outer_trip": int(outer[2]), "expected": expected,
            "loops": [(n, int(t)) for _, n, t in loops]}


def make_inputs(seed, M, K, N):
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
    B = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    # Force the extrema and both signs in structured places, so every build
    # exercises -128*-128, 127*-128, 127*127 and a K-long run of each:
    A[0, :] = -128
    A[1, :] = 127
    B[:, 0] = -128
    B[:, 1] = 127
    A[2, 0] = -128
    A[2, 1] = 127
    B[0, 2] = 127
    B[1, 2] = -128
    ref = A.astype(np.int64) @ B.astype(np.int64)
    assert ref.min() >= -2**31 and ref.max() < 2**31, "reference overflows int32"
    C = ref.astype(np.int32)
    assert A.min() == -128 and A.max() == 127 and B.min() == -128 and B.max() == 127
    return A, B, C


TB = r'''// E1 testbench for Allo's library systolic GEMM (generated by e1_allo_gen.py).
// Reads the numpy-generated int8 inputs and int32 reference for seeds @SEEDS@,
// runs the kernel once per seed, and fails (non-zero exit, so csim_design /
// cosim_design fail) on any file-size, extrema or value mismatch.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include "kernel.h"

static const int M = @M@, K = @K@, N = @N@;

static bool read_file(const char *name, void *dst, size_t bytes) {
  FILE *f = fopen(name, "rb");
  if (!f) { printf("E1 TB ERROR cannot open %s\n", name); return false; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if ((size_t)sz != bytes) {
    printf("E1 TB ERROR %s has %ld bytes, expected %zu\n", name, sz, bytes);
    fclose(f);
    return false;
  }
  size_t got = fread(dst, 1, bytes, f);
  fclose(f);
  if (got != bytes) { printf("E1 TB ERROR short read on %s\n", name); return false; }
  return true;
}

int main() {
  const int seeds[] = {@SEED_LIST@};
  const int nseeds = sizeof(seeds) / sizeof(seeds[0]);
  int failures = 0;
  for (int si = 0; si < nseeds; si++) {
    int seed = seeds[si];
    std::vector<int8_t> A((size_t)M * K), B((size_t)K * N);
    std::vector<int32_t> C((size_t)M * N, 0), Cref((size_t)M * N);
    char fa[64], fb[64], fc[64];
    snprintf(fa, sizeof fa, "A%d.bin", seed);
    snprintf(fb, sizeof fb, "B%d.bin", seed);
    snprintf(fc, sizeof fc, "Cref%d.bin", seed);
    if (!read_file(fa, A.data(), A.size()) || !read_file(fb, B.data(), B.size()) ||
        !read_file(fc, Cref.data(), Cref.size() * sizeof(int32_t))) {
      failures++;
      continue;
    }
    int amin = 127, amax = -128, bmin = 127, bmax = -128;
    for (size_t i = 0; i < A.size(); i++) { if (A[i] < amin) amin = A[i]; if (A[i] > amax) amax = A[i]; }
    for (size_t i = 0; i < B.size(); i++) { if (B[i] < bmin) bmin = B[i]; if (B[i] > bmax) bmax = B[i]; }
    printf("E1 TB seed=%d A[min,max]=[%d,%d] B[min,max]=[%d,%d]\n", seed, amin, amax, bmin, bmax);
    if (amin != -128 || amax != 127 || bmin != -128 || bmax != 127) {
      printf("E1 TB seed=%d FAIL inputs do not span the int8 extrema\n", seed);
      failures++;
      continue;
    }
    // Poison C so a kernel that never writes an element cannot pass.
    for (size_t i = 0; i < C.size(); i++) C[i] = (int32_t)0x7eadbeef;
    gemm(A.data(), B.data(), C.data());
    long long max_err = 0, mism = 0;
    for (size_t i = 0; i < C.size(); i++) {
      long long d = (long long)C[i] - (long long)Cref[i];
      if (d < 0) d = -d;
      if (d != 0) mism++;
      if (d > max_err) max_err = d;
    }
    printf("E1 TB seed=%d %s max_abs_err=%lld mismatches=%lld of %zu\n", seed,
           mism == 0 ? "PASS" : "FAIL", max_err, mism, C.size());
    if (mism != 0) failures++;
  }
  if (failures == 0) printf("E1 TB ALL PASS seeds=%d\n", nseeds);
  else printf("E1 TB FAILED seeds_failed=%d\n", failures);
  return failures == 0 ? 0 : 1;
}
'''

# Allo's backend/hls.py codegen_tcl template, reproduced line for line for the
# parts it emits (project/solution/top/files/part/clock), with the four
# stages Allo's mode string "csim|csynth|cosim|impl" selects and a timer
# around each. Allo's HLSModule cannot run cosim/impl on this host (its
# vitis_hls platform only accepts csim/csyn/sw_emu/hw_emu/hw and its
# vivado_hls platform needs a vivado_hls binary), so the driver invokes
# vitis_hls on this file itself.
TCL = r'''# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#=============================================================================
# run.tcl (E1: Allo codegen_tcl template + timed csim/csynth/cosim/impl)
#=============================================================================
proc e1_stage {name body} {
  set t0 [clock milliseconds]
  puts "E1_STAGE $name START $t0"
  set rc [catch {uplevel 1 $body} err]
  set t1 [clock milliseconds]
  if {$rc} {
    puts "E1_STAGE $name FAILED $t1 ELAPSED_S [expr {($t1 - $t0) / 1000.0}] ERR $err"
  } else {
    puts "E1_STAGE $name END $t1 ELAPSED_S [expr {($t1 - $t0) / 1000.0}]"
  }
  return $rc
}
# Project name
set hls_prj out.prj

# Open/reset the project
open_project ${hls_prj} -reset

open_solution -reset solution1 -flow_target vivado

# Top function of the design is "gemm"
set_top gemm

# Add design and testbench files
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=gnu++0x"
@TB_DATA_FILES@
open_solution "solution1"

# Target device is u280
set_part {@PART@}

# Target frequency
create_clock -period @PERIOD@

# Vivado reports for the export run (utilization, timing summary, route status)
config_export -vivado_report_level 2 -rtl verilog

# Run HLS
set csim_rc [e1_stage csim { csim_design -O }]
set csynth_rc [e1_stage csynth { csynth_design }]
if {$csynth_rc == 0} {
  e1_stage cosim { cosim_design -rtl verilog -trace_level none }
  e1_stage export_impl { export_design -flow impl -rtl verilog -format ip_catalog }
}
puts "E1_TCL_DONE"
exit
'''


def main():
    S = int(sys.argv[1])
    workload = sys.argv[2]
    out = os.path.abspath(sys.argv[3])
    M = K = N = S if workload == "tile" else int(workload)
    assert M % S == 0 and N % S == 0
    os.makedirs(out, exist_ok=True)

    s = build_schedule(M, K, N, S)
    plain = s.build("vhls")  # the un-postprocessed HLS code, kept for inspection
    with open(os.path.join(out, "kernel_plain.cpp"), "w") as f:
        f.write(str(plain))
    s.build(target="vitis_hls", mode="csyn", project=out)
    with open(os.path.join(out, "gemm.mlir"), "w") as f:
        f.write(str(s.module))
    code = open(os.path.join(out, "kernel.cpp")).read()
    # Vitis HLS refuses to co-simulate an m_axi pointer port without depth=
    # ("A depth specification is required for MAXI interface port ... for
    # cosimulation"); Allo emits none because its own path is v++ hw_emu/hw.
    # depth on m_axi is a simulation-only attribute, so add the element count
    # of each argument (A: M*K, B: K*N, C: M*N, in kernel argument order) to
    # a copy; kernel_allo.cpp keeps Allo's verbatim output for the diff.
    with open(os.path.join(out, "kernel_allo.cpp"), "w") as f:
        f.write(code)
    depths = iter([M * K, K * N, M * N])
    patched = []
    for ln in code.splitlines():
        m = re.match(r"(\s*#pragma HLS interface m_axi port=\w+ offset=slave bundle=\w+)\s*$", ln)
        if m:
            ln = f"{m.group(1)} depth={next(depths)}"
        patched.append(ln)
    code = "\n".join(patched) + "\n"
    # (count m_axi pragmas only: the PE stream pragmas carry depth= as well)
    n_maxi_depth = len(re.findall(r"#pragma HLS interface m_axi port=\w+ offset=slave bundle=\w+ depth=\d+", code))
    assert n_maxi_depth == 3, f"expected exactly three m_axi pragmas with depth, got {n_maxi_depth}"
    with open(os.path.join(out, "kernel.cpp"), "w") as f:
        f.write(code)

    ok, detail = check_output_stationary(code)
    tiles = tile_loop_info(code, M, N, S)
    m_axi = re.findall(r"#pragma HLS interface m_axi port=(\w+) offset=slave bundle=(\w+)", code)
    info = {
        "S": S, "M": M, "K": K, "N": N, "workload": workload,
        "top": "gemm", "library_function": "systolic_gemm",
        "output_stationary": ok, "pe": detail,
        "pe_count_expected": S * S,
        "tile_loops": tiles,
        "dataflow_pragmas": code.count("#pragma HLS dataflow"),
        "stream_decls": len(re.findall(r"hls::stream< int8_t > \w+\[", code)),
        "m_axi_ports": [{"port": p, "bundle": b} for p, b in m_axi],
        "period_ns": PERIOD_NS, "part": PART,
    }
    print("E1 GEN", json.dumps(info, indent=1))
    if not ok:
        print("E1 GEN ERROR: emitted PE is not output-stationary:", detail)
        sys.exit(2)
    if detail["pe_count"] != S * S:
        print(f"E1 GEN ERROR: {detail['pe_count']} PE functions, expected {S*S}")
        sys.exit(2)
    if "error" in tiles or tiles["outer_trip"] != tiles["expected"]:
        print("E1 GEN ERROR: tile loop check failed", tiles)
        sys.exit(2)

    data_files = []
    for seed in SEEDS:
        A, B, C = make_inputs(seed, M, K, N)
        for name, arr in (("A", A), ("B", B), ("Cref", C)):
            fn = f"{name}{seed}.bin"
            arr.tofile(os.path.join(out, fn))
            data_files.append(fn)
    tb = (TB.replace("@SEEDS@", str(list(SEEDS))).replace("@M@", str(M))
          .replace("@K@", str(K)).replace("@N@", str(N))
          .replace("@SEED_LIST@", ", ".join(str(x) for x in SEEDS)))
    with open(os.path.join(out, "tb.cpp"), "w") as f:
        f.write(tb)
    tcl = (TCL.replace("@TB_DATA_FILES@", "\n".join(f"add_files -tb {fn}" for fn in data_files))
           .replace("@PART@", PART).replace("@PERIOD@", PERIOD_NS))
    with open(os.path.join(out, "run_e1.tcl"), "w") as f:
        f.write(tcl)
    with open(os.path.join(out, "e1_gen_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print("E1 GEN OK", out)


if __name__ == "__main__":
    main()
