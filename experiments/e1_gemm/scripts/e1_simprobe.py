# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the OpenMP simulator's time goes, and how it scales with a longer workload.

    python3 e1_simprobe.py profile <size>          # gemm8: cProfile of the build, rusage of two runs
    python3 e1_simprobe.py tiles <size> <W> <dir>  # gemmtiles-W on the simulator and Vitis csim

`tiles` takes REORDER=1, which moves the `frame` calls ahead of the cells in
the generated kernel.cpp before Vitis's C simulation (see SIM_TIME.md), and
VITIS_ONLY=1, which skips the simulator.
"""
import cProfile, io, json, os, pstats, re, resource, subprocess, sys, time
import numpy as np
import allo.spmw as spmw


def usage():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime, r.ru_stime, r.ru_nvcsw, r.ru_nivcsw


def timed(call):
    u0, s0, v0, i0 = usage()
    t0 = time.time()
    call()
    t1 = time.time()
    u1, s1, v1, i1 = usage()
    return {
        "wall": round(t1 - t0, 3),
        "user": round(u1 - u0, 2),
        "sys": round(s1 - s0, 2),
        "vol_ctx": v1 - v0,
        "invol_ctx": i1 - i0,
    }


def profile(size):
    from test_spmw_gemm_int8 import gemm_int8_of

    rng = np.random.default_rng(3)
    a = rng.integers(-128, 128, (size, size)).astype(np.int8)
    b = rng.integers(-128, 128, (size, size)).astype(np.int8)
    g = gemm_int8_of(size)
    prof = cProfile.Profile()
    t0 = time.time()
    prof.enable()
    mod = spmw.build(g, target="simulator")
    prof.disable()
    build = time.time() - t0
    out = io.StringIO()
    pstats.Stats(prof, stream=out).sort_stats("cumulative").print_stats(28)
    lines = [l for l in out.getvalue().splitlines() if re.match(r"\s+\d", l)]
    runs = []
    for _ in range(2):
        c = np.zeros((size, size), np.int32)
        runs.append(timed(lambda: mod(a, b, c)))
        runs[-1]["pass"] = bool(
            np.array_equal(c, a.astype(np.int32) @ b.astype(np.int32))
        )
    print(json.dumps({"size": size, "build_s": round(build, 2), "runs": runs}))
    print("\n".join(lines))


TB = r"""#include <cstdint>
#include <cstdio>
#include <vector>
#include "kernel.h"
int main() {
  FILE *f = fopen(OPERANDS, "rb");
  int32_t h[4];
  if (!f || fread(h, 4, 4, f) != 4) { printf("NO OPERANDS\n"); return 2; }
  int na = h[0], nb = h[1], nc = h[2], n = h[3];
  std::vector<int8_t> at(na * n), bt(nb * n);
  std::vector<int32_t> want(nc * n), ct(nc * n, 0);
  fread(at.data(), 1, at.size(), f); fread(bt.data(), 1, bt.size(), f);
  fread(want.data(), 4, want.size(), f); fclose(f);
  top(at.data(), bt.data(), ct.data());
  int bad = 0;
  for (size_t i = 0; i < ct.size(); i++) bad += ct[i] != want[i];
  printf(bad ? "TILES CSIM FAIL %d\n" : "TILES CSIM PASS %d\n", bad);
  return bad != 0;
}
"""
TCL = """open_project csim_prj -reset
set_top top
add_files kernel.cpp
add_files -tb tb.cpp
open_solution -reset sol -flow_target vivado
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.333
csim_design
exit
"""
LIBS = ":".join(
    "/opt/xilinx/Vitis_HLS/2023.2/lnx64/" + p
    for p in (
        "tools/fpo_v7_1",
        "tools/fft_v9_1",
        "tools/fir_v7_0",
        "tools/dds_v6_0",
        "csim",
    )
)


def tiles(size, side, out):
    from test_spmw_gemm_tiles import tiles_workload

    g = tiles_workload(size, side)
    at, bt = g.spmw_operands["At"], g.spmw_operands["Bt"]
    want = g.spmw_expected["Ct"]
    steps = g.spmw_tiles["steps"]
    res = {"size": size, "W": side, "steps_a_cell": steps}
    t0 = time.time()
    mod = None if os.environ.get("VITIS_ONLY") else spmw.build(g, target="simulator")
    res["omp_build_s"] = round(time.time() - t0, 2)
    runs = []
    for _ in range(2 if mod is not None else 0):
        ct = np.zeros(want.shape, np.int32)
        runs.append(timed(lambda: mod(at, bt, ct)))
        runs[-1]["pass"] = bool(np.array_equal(ct, want))
    res["omp_runs"] = runs
    del mod
    t0 = time.time()
    g2 = tiles_workload(size, side)
    spmw.build(g2, target="vitis_hls", mode="csim", project=out)
    res["vitis_codegen_s"] = round(time.time() - t0, 2)
    if os.environ.get("REORDER"):
        # Vitis's C simulation runs a dataflow region's processes one after
        # another in the order they are called, and SPMW calls the cells
        # before the `frame` units that feed them. Call order means nothing to
        # the hardware; move the frames ahead of the first cell.
        kpath = os.path.join(out, "kernel.cpp")
        lines = open(kpath).read().split("\n")
        start = next(i for i, l in enumerate(lines) if l.startswith("void top("))
        frames = [
            i
            for i in range(start, len(lines))
            if lines[i].lstrip().startswith("frame_r0(")
        ]
        first_pe = next(
            i for i in range(start, len(lines)) if re.match(r"\s+pe_r\d+\(", lines[i])
        )
        moved = [lines[i] for i in frames]
        keep = [l for i, l in enumerate(lines) if i not in set(frames)]
        at_pe = keep.index(lines[first_pe])
        keep[at_pe:at_pe] = moved
        open(kpath, "w").write("\n".join(keep))
        res["reordered_frames"] = len(moved)
    with open(os.path.join(out, "operands.bin"), "wb") as f:
        f.write(
            np.array(
                [at.shape[0], bt.shape[0], want.shape[0], size], np.int32
            ).tobytes()
        )
        f.write(np.ascontiguousarray(at).tobytes())
        f.write(np.ascontiguousarray(bt).tobytes())
        f.write(np.ascontiguousarray(want).tobytes())
    path = os.path.abspath(os.path.join(out, "operands.bin"))
    open(os.path.join(out, "tb.cpp"), "w").write('#define OPERANDS "%s"\n' % path + TB)
    open(os.path.join(out, "csim.tcl"), "w").write(TCL)
    t0 = time.time()
    done = subprocess.run(
        ["vitis_hls", "-f", "csim.tcl"], cwd=out, capture_output=True, text=True
    )
    res["vitis_wall_s"] = round(time.time() - t0, 2)
    m = re.search(
        r"Finished Command csim_design.*?Elapsed time: ([\d.]+) seconds", done.stdout
    )
    res["csim_design_s"] = float(m.group(1)) if m else None
    res["vitis_pass"] = "TILES CSIM PASS" in done.stdout
    build = os.path.join(out, "csim_prj", "sol", "csim", "build")
    env = dict(
        os.environ, LD_LIBRARY_PATH=os.environ.get("LD_LIBRARY_PATH", "") + ":" + LIBS
    )
    hot = []
    for _ in range(2):
        t0 = time.time()
        r = subprocess.run(
            ["./csim.exe"], cwd=build, env=env, capture_output=True, text=True
        )
        hot.append(round(time.time() - t0, 4))
        res["vitis_hot_pass"] = "TILES CSIM PASS" in r.stdout
    res["vitis_hot_s"] = hot
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "profile":
        profile(int(sys.argv[2]))
    else:
        tiles(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
