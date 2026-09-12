#!/usr/bin/env python3
"""E4: the old weight image against the new one, on the files the runs used.

    compare_images.py <old run dir> <new run dir> [...]

For each pair it replays the feed rule that the hardware in that run implements
-- one PE a cycle over N^3 rows, or one row of PEs a cycle over N^2 -- against
the run's own weights.hex and wrow0.hex, and checks that the two recover the
same N^2 x N array of PE files, byte for byte, and that both agree with the
pefiles.hex the testbench dumps the PEs against. The point of the change is
that the same weights land in the same slots; this is where that is checked on
the images themselves rather than inferred from a passing simulation.
"""

import json
import os
import sys

import numpy as np


def read_rows(path, N):
    """One hex word a line, byte j at bits [8j +: 8] -- write_rows_hex's inverse."""
    with open(path, encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    out = np.zeros((len(words), N), dtype=np.uint8)
    for i, w in enumerate(words):
        v = int(w, 16)
        for j in range(N):
            out[i, j] = (v >> (8 * j)) & 0xFF
    return out


def replay(run):
    """The PE files [N^2, N] the feed in this run's images actually delivers."""
    meta = json.load(open(os.path.join(run, "meta.json"), encoding="utf-8"))
    N, loader, L = meta["N"], meta["loader"], meta["WLEN"]
    rows = read_rows(os.path.join(run, "weights.hex"), N)
    stale = read_rows(os.path.join(run, "wrow0.hex"), N)[0]
    # the testbench writes the real row 0 through the port as the fill's last
    # write, so the SRAM holds wrow0 at address 0 and the read register holds
    # what was there before it -- weights.hex row 0 -- at the feed's first cycle
    mem = rows.copy()
    stale, mem[0] = mem[0].copy(), stale
    files = np.zeros((N * N, N), dtype=np.uint8)
    for c in range(L):
        reg = stale if c == 0 else mem[c - 1]
        if loader == "pe":
            q = c % (N * N)
            k = c // (N * N)
            if k < N:
                files[q, k] = reg[q // N]
        else:
            r = c % N
            k = c // N
            if k < N:
                for col in range(N):
                    files[N * col + r, k] = reg[col]
    return meta, files, rows


def main():
    runs = sys.argv[1:]
    assert len(runs) % 2 == 0 and runs, "give pairs: <old> <new> [<old> <new> ...]"
    ok = True
    for i in range(0, len(runs), 2):
        old, new = runs[i], runs[i + 1]
        mo, fo, ro = replay(old)
        mn, fn, rn = replay(new)
        N = mo["N"]
        assert mn["N"] == N and mo["loader"] == "pe" and mn["loader"] == "row"
        # the testbench's own copy of what each PE must hold
        want = np.array(
            [int(x, 16) for x in open(os.path.join(old, "pefiles.hex"), encoding="utf-8")],
            dtype=np.uint8,
        ).reshape(N * N, N)
        want_new = np.array(
            [int(x, 16) for x in open(os.path.join(new, "pefiles.hex"), encoding="utf-8")],
            dtype=np.uint8,
        ).reshape(N * N, N)
        checks = {
            "old image delivers pefiles.hex": np.array_equal(fo, want),
            "new image delivers pefiles.hex": np.array_equal(fn, want_new),
            "the two runs want the same weights": np.array_equal(want, want_new),
            "same weights in the same slots": np.array_equal(fo, fn),
            "the new image is N times shorter": rn.shape[0] * N == ro.shape[0],
            "the weights are not all zero": int(np.count_nonzero(fn)) > 0,
        }
        bad = [k for k, v in checks.items() if not v]
        ok = ok and not bad
        print(
            f"N={N:<3} old {ro.shape[0]:>5} rows x {N} B (feed {mo['WLEN']:>5} cy)"
            f"  new {rn.shape[0]:>5} rows x {N} B (feed {mn['WLEN']:>4} cy)"
            f"  distinct bytes {int(np.count_nonzero(fn)):>5}/{N**3}"
            f"  -> {'OK' if not bad else 'MISMATCH: ' + '; '.join(bad)}"
        )
        # where the two images differ, and where they cannot
        lanes_old = int((ro != 0).sum(axis=1).max())
        lanes_new = int((rn != 0).sum(axis=1).max())
        print(f"     live byte lanes per row: old <= {lanes_old}, new <= {lanes_new} (of {N})")
    print("IMAGE COMPARISON", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
