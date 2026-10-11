#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ws_allo_gen.py's array with each cell's weight arriving through a FIFO.

usage: ws_probe2.py S OUTDIR

A variant kept to answer one question, not a measured design.  ws_allo_gen.py
hands every cell the array B and its own indices, and the cell reads W[k, n]
once.  Here the tile copies B into a grid of one-entry buffers (`weight_load`),
each cell reads its own (`W_in[0]`), and the schedule turns the grid into
FIFOs with the call that makes the other two.  Everything else is that
script's: the same cell loop, loaders, drain, partitioning and pipelining.

It runs the unscheduled design on Allo's LLVM backend against numpy, emits the
Vitis kernel into OUTDIR and stops; it writes no testbench.  probe32.sh
synthesised it at 32x32 to see whether the way a cell gets its weight was what
kept Vitis HLS in its optimisation phase at that size.
"""
import json
import os
import re
import sys

import numpy as np

import allo
from allo import dsl
from allo.ir.types import index, int8, int32
from allo.ir.utils import MockBuffer

SEEDS = (0, 1, 2)


def PE_kernel_ws[
    TyA, TyB, TyC, M: int32, Kt: int32, Nt: int32
](A_in: "TyA[M]", P_in: "TyC[M]", W_in: "TyB[1]", A_out: "TyA[M]", P_out: "TyC[M]",):
    w: TyB = W_in[0]
    for m in dsl.grid(M, name="step"):
        a: TyA = A_in[m]
        p: TyC = P_in[m]
        P_out[m] = p + a * w
        A_out[m] = a


def systolic_tile_ws[
    TyA, TyB, TyC, M: int32, Kt: int32, Nt: int32
](A: "TyA[M, Kt]", B: "TyB[Kt, Nt]", C: "TyC[M, Nt]"):
    A_fifo: TyA[Kt, Nt + 1, M]
    P_fifo: TyC[Nt, Kt + 1, M]
    W_fifo: TyB[Kt, Nt, 1]
    A_drain: TyA[Kt]

    for wk in range(Kt, name="weight_load"):
        for wn in range(Nt):
            W_fifo[wk, wn, 0] = B[wk, wn]
    for m in range(M, name="data_load"):
        for k in range(Kt):
            A_fifo[k, 0, m] = A[m, k]
        for n in range(Nt):
            P_fifo[n, 0, m] = 0
    for k, n in dsl.grid(Kt, Nt, name="PE"):
        PE_kernel_ws[TyA, TyB, TyC, M, Kt, Nt](
            A_fifo[k, n], P_fifo[n, k], W_fifo[k, n], A_fifo[k, n + 1], P_fifo[n, k + 1]
        )
    for m in range(M, name="data_drain"):
        for k in range(Kt):
            A_drain[k] = A_fifo[k, Nt, m]
        for n in range(Nt):
            C[m, n] = P_fifo[n, Kt, m]


def systolic_ws[
    TyA, TyB, TyC, M: int32, Kt: int32, Nt: int32
](A: "TyA[M, Kt]", B: "TyB[Kt, Nt]", C: "TyC[M, Nt]"):
    local_A: TyA[M, Kt]
    local_B: TyB[Kt, Nt]
    local_C: TyC[M, Nt]

    for am, ak in dsl.grid(M, Kt, name="load_A"):
        local_A[am, ak] = A[am, ak]
    for bk, bn in dsl.grid(Kt, Nt, name="load_B"):
        local_B[bk, bn] = B[bk, bn]
    systolic_tile_ws[TyA, TyB, TyC, M, Kt, Nt](local_A, local_B, local_C)
    for sm, sn in dsl.grid(M, Nt, name="store_C"):
        C[sm, sn] = local_C[sm, sn]


def schedule_ws(s, depth):
    """The library's schedule, with the roles of the three arrays exchanged."""
    s.partition(s.local_B, dim=0)  # a weight a cell, each read by its own
    s.partition(s.local_A, dim=2)  # a step's activations, side by side
    s.partition(s.local_C, dim=2)  # a step's results, side by side
    loops = s.get_loops(s.top_func_name)
    s.pipeline(loops["load_A"]["am"])
    s.pipeline(loops["load_B"]["bk"])
    s.pipeline(loops["store_C"]["sm"])
    s.pipeline(s.get_loops("PE_kernel_ws")["step"]["m"])
    s.pipeline(s.get_loops("systolic_tile_ws")["weight_load"]["wk"])
    pe = s.unfold("systolic_tile_ws:PE", [0, 1])
    s.to(MockBuffer("systolic_tile_ws", "A_fifo"), pe, axis=1, depth=depth)
    s.to(MockBuffer("systolic_tile_ws", "P_fifo"), pe, axis=0, depth=depth)
    s.to(MockBuffer("systolic_tile_ws", "W_fifo"), pe, axis=0, depth=2)
    return s


def build_schedule(S):
    args = [int8, int8, int32, S, S, S]
    inner = schedule_ws(allo.customize(systolic_ws, instantiate=args), S + 1)

    def gemm(A: int8[S, S], B: int8[S, S], C: int32[S, S]):
        systolic_ws[int8, int8, int32, S, S, S, "gemm"](A, B, C)

    s = allo.customize(gemm)
    s.compose(inner, id="gemm")
    return s


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
                return code[start : j + 1]
    raise ValueError(name)


def check_weight_stationary(code, S):
    """Every cell reads its weight once, before its loop, and stores nothing.

    Returns (ok, detail).  The weight is the cell's one two-dimensional
    argument; a store to an array argument would make it something else.
    """
    names = re.findall(r"^void (PE_kernel_ws\w*)\(", code, flags=re.M)
    if len(names) != S * S:
        return False, {"error": f"{len(names)} cell functions, expected {S * S}"}
    detail = {"pe_count": len(names), "pe_function": names[0]}
    for name in names:
        body = function_body(code, name)
        header = body[: body.index("{")]
        weight = re.search(r"int8_t (\w+)\[\d+\]\[\d+\]", header)
        if weight is None:
            return False, {"error": f"{name} has no weight argument"}
        reads, stores, depth = [], 0, 0
        for line in body[body.index("{") :].splitlines():
            if re.search(rf"= {weight.group(1)}\[", line):
                reads.append(depth)
            if re.match(r"\s*\w+\[[^\]]*\](\[[^\]]*\])* = ", line):
                stores += 1
            depth += line.count("{") - line.count("}")
        if reads != [1] or stores:
            return False, {"error": name, "weight_reads": reads, "stores": stores}
    loop = re.search(r"for \(int (\w+) = 0; \1 < (\d+); \1\+\+\)", body)
    detail["step_loop_trip"] = int(loop.group(2)) if loop else None
    detail["pipeline_pragma"] = "#pragma HLS pipeline II=1" in body
    return loop is not None, detail


def make_inputs(seed, M, K, N):
    """E1's operands: random int8 with the extrema planted."""
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
    B = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    A[0, :] = -128
    A[1, :] = 127
    B[:, 0] = -128
    B[:, 1] = 127
    A[2, 0] = -128
    A[2, 1] = 127
    B[0, 2] = 127
    B[1, 2] = -128
    ref = A.astype(np.int64) @ B.astype(np.int64)
    assert ref.min() >= -(2**31) and ref.max() < 2**31, "reference overflows int32"
    return A, B, ref.astype(np.int32)


TB = r"""// Testbench for Allo's weight-stationary systolic GEMM (ws_allo_gen.py): E1's.
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
  if (!f) { printf("WS TB ERROR cannot open %s\n", name); return false; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if ((size_t)sz != bytes) {
    printf("WS TB ERROR %s has %ld bytes, expected %zu\n", name, sz, bytes);
    fclose(f);
    return false;
  }
  size_t got = fread(dst, 1, bytes, f);
  fclose(f);
  if (got != bytes) { printf("WS TB ERROR short read on %s\n", name); return false; }
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
    printf("WS TB seed=%d A[min,max]=[%d,%d] B[min,max]=[%d,%d]\n", seed, amin, amax, bmin, bmax);
    if (amin != -128 || amax != 127 || bmin != -128 || bmax != 127) {
      printf("WS TB seed=%d FAIL inputs do not span the int8 extrema\n", seed);
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
    printf("WS TB seed=%d %s max_abs_err=%lld mismatches=%lld of %zu\n", seed,
           mism == 0 ? "PASS" : "FAIL", max_err, mism, C.size());
    if (mism != 0) failures++;
  }
  if (failures == 0) printf("WS TB ALL PASS seeds=%d\n", nseeds);
  else printf("WS TB FAILED seeds_failed=%d\n", failures);
  return failures == 0 ? 0 : 1;
}
"""

# Allo emits the kernel inside `extern "C"`, so the testbench must see it so.
HEADER = """#ifndef KERNEL_H
#define KERNEL_H
#include <stdint.h>
extern "C" {
void gemm(int8_t *A, int8_t *B, int32_t *C);
}
#endif
"""


def main():
    S = int(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    os.makedirs(out, exist_ok=True)

    # 1. The design as written, on the LLVM backend.
    plain = allo.customize(systolic_ws, instantiate=[int8, int8, int32, S, S, S])
    run = plain.build()
    for seed in SEEDS:
        A, B, ref = make_inputs(seed, S, S, S)
        C = np.zeros((S, S), dtype=np.int32)
        run(A, B, C)
        np.testing.assert_array_equal(C, ref)
    print(f"WS GEN llvm matches numpy for seeds {list(SEEDS)}")

    # 2. Schedule and emit.
    s = build_schedule(S)
    with open(os.path.join(out, "kernel_plain.cpp"), "w") as f:
        f.write(str(s.build("vhls")))
    s.build(target="vitis_hls", mode="csyn", project=out)
    with open(os.path.join(out, "gemm.mlir"), "w") as f:
        f.write(str(s.module))
    code = open(os.path.join(out, "kernel.cpp")).read()
    with open(os.path.join(out, "kernel_allo.cpp"), "w") as f:
        f.write(code)
    # Cosimulation needs depth= on an m_axi pointer port (E1's generator says
    # why); it is the element count of each argument, in argument order.
    depths = iter([S * S, S * S, S * S])
    patched = []
    for ln in code.splitlines():
        m = re.match(
            r"(\s*#pragma HLS interface m_axi port=\w+ offset=slave bundle=\w+)\s*$", ln
        )
        patched.append(f"{m.group(1)} depth={next(depths)}" if m else ln)
    code = "\n".join(patched) + "\n"
    n_depth = len(re.findall(r"m_axi port=\w+ offset=slave bundle=\w+ depth=\d+", code))
    assert n_depth == 3, f"expected three m_axi pragmas with depth, got {n_depth}"
    with open(os.path.join(out, "kernel.cpp"), "w") as f:
        f.write(code)
    with open(os.path.join(out, "kernel.h"), "w") as f:
        f.write(HEADER)

    print("PROBE EMITTED", out)


if __name__ == "__main__":
    main()
