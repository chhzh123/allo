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

SPMW = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1 else "/scratch/hc676/spmw_fft_rolled"
)
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
    ii_table()
    area_table()


# A pipelined loop's line in a Vitis csynth report: the columns are
# name | type | slack | latency | latency(ns) | iteration latency | INTERVAL |
# trip count | pipelined | ...
LOOP = re.compile(r"^\s*\|\s+o\s+(\S+)\s*\|([^|]*\|){5}\s*(\d+)\|\s*(\d+)\|\s*(\S+)\|")


def ii_table():
    """Initiation intervals, read out of the reports rather than inferred.

    A cycle count can look like II=1 for several reasons; the report says it
    or it does not. The worst II over every pipelined loop of every role is
    what the array's throughput actually rests on.
    """
    print("\nII, from each role's csynth report (worst loop per role)\n")
    print(f"{'W':>3} {'roles':>6} {'worst II':>9} {'longest loop':>44} {'trip':>8}")
    print("-" * 75)
    for lanes in (1, 2, 4, 8, 16):
        root = SPMW / f"n{N}w{lanes}_c"
        if not root.is_dir():
            print(f"{lanes:>3} {'not built':>6}")
            continue
        worst, longest, roles = 0, None, 0
        for role in sorted(root.glob("*_r*/prj/sol/syn/report/csynth.rpt")):
            roles += 1
            for line in role.read_text(errors="replace").splitlines():
                m = LOOP.match(line)
                if not m:
                    continue
                name, _, ii, trip = (
                    m.group(1),
                    m.group(2),
                    int(m.group(3)),
                    int(m.group(4)),
                )
                worst = max(worst, ii)
                if longest is None or trip > longest[2]:
                    longest = (name, ii, trip)
        if longest is None:
            print(f"{lanes:>3} {roles:>6} {'no loops':>9}")
            continue
        print(
            f"{lanes:>3} {roles:>6} {worst:>9} {longest[0][-44:]:>44} {longest[2]:>8}"
        )


def area_table():
    """Routed resources for the array, from the place-and-route reports."""
    print("\nRouted on xcu280-fsvh2892-2L-e at 3.333 ns\n")
    print(
        f"{'W':>3} {'LUT':>8} {'FF':>8} {'DSP':>6} {'BRAM18':>7} {'WNS ns':>8} {'MHz':>5}"
    )
    print("-" * 50)
    for lanes in (1, 2, 4, 8, 16):
        log = SPMW / f"n{N}w{lanes}_p.log"
        util = SPMW / f"n{N}w{lanes}_p" / "util.rpt"
        if not util.is_file():
            print(f"{lanes:>3} {'not measured':>8}")
            continue
        txt = util.read_text(errors="replace")

        def cell(name, txt=txt):
            m = re.search(r"\|\s*" + re.escape(name) + r"\s*\|\s*([\d,]+)\s*\|", txt)
            return int(m.group(1).replace(",", "")) if m else None

        wns = None
        if log.is_file():
            m = re.search(r"ARRAY WNS (-?[\d.]+)", log.read_text(errors="replace"))
            wns = float(m.group(1)) if m else None
        b18 = 2 * (cell("RAMB36/FIFO") or 0) + (cell("RAMB18") or 0)
        mhz = f"{1000 / (3.333 - wns):.0f}" if wns is not None else "-"
        print(
            f"{lanes:>3} {cell('CLB LUTs') or cell('Slice LUTs'):>8} "
            f"{cell('CLB Registers') or cell('Slice Registers'):>8} "
            f"{cell('DSPs'):>6} {b18:>7} "
            f"{wns if wns is not None else '-':>8} {mhz:>5}"
        )


if __name__ == "__main__":
    main()
