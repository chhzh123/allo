# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthesis-time scaling harness for the rolled ``spmw.map`` systolic top.

Emits the IR-driven rolled top (``spmw.build(target="rolled")``) for a sweep of square grid sizes,
runs a real Vitis HLS ``csynth_design`` on each, and records the synthesis wall-clock alongside the
distinct-role-body count. The load-bearing synthesis-time-win claim (AC-4) is that the rolled map
lets HLS synthesize O(#roles) function bodies, not O(P0*P1): the body count stays constant while the
grid grows, so the csynth front end schedules a fixed number of modules and only replicates the one
already-scheduled interior body across the mesh.

Run on the remote host with the Vitis env sourced (see CLAUDE.md):

    python3 scripts/spmw_synth_scaling.py 4 8 16 --out docs/spmw_synth_scaling_data.md

Each size is synthesized in its own temp project; a per-size timeout caps a run that does not finish
and is reported as ``TIMEOUT`` rather than silently dropped.
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

import allo.spmw as spmw
from allo.ir.types import float32

# Contraction depth is held fixed so the interior body is identical across grid sizes -- this
# isolates the grid-replication effect the harness is measuring from any K-driven body growth.
_K = 4


def _square_twin(n):
    """A square ``n x n`` systolic GEMM twin with a fixed contraction depth ``_K``."""
    grid = spmw.mesh((n, n))

    @spmw.unit
    def pe(ctx):
        c: float32 = 0
        for k in range(_K):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: float32[n, _K], B: float32[_K, n], C: float32[n, n]):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    return gemm


def _distinct_bodies(kernel_cpp):
    """The number of distinct function bodies HLS sees in the emitted top (O(#roles), not O(P^2))."""
    return len(set(re.findall(r"\bvoid\s+(\w+)\s*\(", kernel_cpp)))


def _kill_group(proc):
    """Kill the process group led by ``proc`` (the Vitis loader wrapper + its grandchildren)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def _run_csynth(project, timeout):
    """Run ``vitis_hls -f run.tcl`` in ``project``; return (elapsed_s, ok, report_dir).

    Vitis launches its real engine as a grandchild via a loader wrapper, so a plain timeout would
    orphan it -- and the orphan keeps writing into the project dir, racing the temp-dir cleanup. Run
    it in its own session and kill the whole process group on timeout so nothing lingers.
    """
    start = time.perf_counter()
    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        ["vitis_hls", "-f", "run.tcl"],
        cwd=project,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        return time.perf_counter() - start, False, None
    elapsed = time.perf_counter() - start
    report = os.path.join(project, "rolled.prj", "solution1", "syn", "report")
    return elapsed, rc == 0 and os.path.isdir(report), report


def measure(sizes, timeout):
    """Synthesize the rolled top at each grid size and yield a result row per size."""
    for n in sizes:
        # ignore_cleanup_errors: a killed/slow Vitis can leave the project dir non-empty at cleanup;
        # a cleanup race must not abort the sweep or lose the already-measured result.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            project = spmw.build(_square_twin(n), target="rolled", project=tmp)
            bodies = _distinct_bodies(project.hls_code)
            elapsed, ok, _ = _run_csynth(tmp, timeout)
            yield {
                "grid": f"{n}x{n}",
                "instances": n * n,
                "bodies": bodies,
                "csynth_s": elapsed,
                "ok": ok,
            }


def _format_table(rows):
    lines = [
        "| grid | PE instances | distinct role bodies | csynth wall-clock (s) | status |",
        "|------|--------------|----------------------|-----------------------|--------|",
    ]
    for r in rows:
        status = "CSYNTH_OK" if r["ok"] else "TIMEOUT/FAIL"
        lines.append(
            f"| {r['grid']} | {r['instances']} | {r['bodies']} | "
            f"{r['csynth_s']:.1f} | {status} |"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sizes", nargs="+", type=int, help="square grid extents, e.g. 8 16 32"
    )
    parser.add_argument(
        "--timeout", type=int, default=3600, help="per-size csynth timeout in seconds"
    )
    parser.add_argument(
        "--out", default=None, help="write the markdown table to this file"
    )
    args = parser.parse_args(argv)

    rows = []
    for row in measure(args.sizes, args.timeout):
        rows.append(row)
        print(
            f"{row['grid']}: {row['bodies']} bodies, "
            f"{row['csynth_s']:.1f}s, {'OK' if row['ok'] else 'TIMEOUT/FAIL'}",
            flush=True,
        )
    table = _format_table(rows)
    print("\n" + table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(table + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
