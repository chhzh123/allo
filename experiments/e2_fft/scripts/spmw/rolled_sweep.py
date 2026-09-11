#!/usr/bin/env python3
"""The rolled FFT's unroll sweep, against HP-FFT at matched unroll factors.

One definition for both sides, the one E2 settled on and HP-FFT's own harness
records: **cycles are rising clock edges; a transform's latency is its last
output beat minus the launch's first input beat; the steady interval is the
median of consecutive completion differences.** The correction E2 records is
kept: the interval is *not* ``(launch total - first output) / 32``, which
starts at the first output beat and divides by the wrong count.

The datapaths are matched here rather than left to be read around. HP-FFT's
boundary is ``hls::stream<hls::vector<complex<float>, UF*2>>`` -- ``2*UF``
complex samples a beat -- and the rolled SPMW design's is ``W`` lanes of one
complex sample a cycle. So **W = 2 * UF** is the matched point, and both sides
then have the same ideal interval, ``N / (2 * UF)``.

usage: rolled_sweep.py [spmw_log_dir] [hpfft_dir]
"""
import json
import pathlib
import re
import statistics
import sys

SPMW = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/hc676/spmw_fft_rolled")
HPFFT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "/scratch/hc676/e2_hpfft")
N = 256


def spmw_row(n, lanes):
    """Parse one rolled-FFT cosim log into the shared definition."""
    log = SPMW / f"n{n}w{lanes}_c.log"
    if not log.is_file():
        return None
    text = log.read_text(errors="replace")
    if "SPMW COSIM PASS" not in text:
        return {"status": "FAIL" if "COSIM" in text else "not run"}
    done, first_in = [], None
    for m in re.finditer(r"SPMW XFORM (\d+) done_cycle=(\d+) first_in=(\d+)", text):
        done.append(int(m.group(2)))
        first_in = int(m.group(3))
    if len(done) < 4:
        return {"status": f"only {len(done)} transform completions reported"}
    diffs = [b - a for a, b in zip(done, done[1:])]
    tail = diffs[len(diffs) // 4 :]  # the last 3/4, as HP-FFT defines it
    return {
        "status": "PASS",
        "first_in": first_in,
        "latency": done[0] - first_in,
        "interval": statistics.median(tail),
        "interval_min": min(diffs),
        "interval_max": max(diffs),
        "transforms": len(done),
        "ideal": n / lanes,
        "samples_per_cycle": lanes,
    }


def hpfft_row(n, uf):
    f = HPFFT / f"n{n}" / f"UF{uf}_s" / "cosim_events.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text())
    return {
        "status": "PASS",
        "first_in": d["first_input_cycle"],
        "latency": d["first_transform_completion_cycles"],
        "interval": d["steady_interval_cycles_median"],
        "interval_min": d["steady_interval_min"],
        "interval_max": d["steady_interval_max"],
        "transforms": d["transforms_completed"],
        "ideal": n / (2 * uf),
        "samples_per_cycle": 2 * uf,
    }


def main():
    hdr = (
        f"{'UF':>4} {'system':<10} {'s/cyc':>6} {'latency':>8} {'interval':>9} "
        f"{'ideal':>7} {'%ideal':>7}  note"
    )
    print(f"N = {N}, matched on samples a cycle (HP-FFT UF k == SPMW W = 2k)\n")
    print(hdr)
    print("-" * len(hdr))
    for uf, lanes in ((0.5, 1), (1, 2), (2, 4), (4, 8), (8, 16)):
        for name, row in (
            ("SPMW W%d" % lanes, spmw_row(N, lanes)),
            ("HP-FFT", hpfft_row(N, int(uf)) if uf >= 1 else None),
        ):
            if row is None:
                print(f"{uf:>4} {name:<10} {'':>6} {'not measured':>8}")
                continue
            if row.get("status") != "PASS":
                print(f"{uf:>4} {name:<10} {'':>6} {row['status']:>8}")
                continue
            pct = 100 * row["ideal"] / row["interval"]
            print(
                f"{uf:>4} {name:<10} {row['samples_per_cycle']:>6} "
                f"{row['latency']:>8} {row['interval']:>9.1f} "
                f"{row['ideal']:>7.0f} {pct:>6.1f}%  "
                f"min {row['interval_min']} max {row['interval_max']}, "
                f"{row['transforms']} transforms"
            )
        print()


if __name__ == "__main__":
    main()
