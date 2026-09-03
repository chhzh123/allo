# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Table 5: lines of code per design.

`cloc` is not installed on the evaluation machine, so this applies cloc's
counting rule directly: a line counts when it is neither blank nor wholly a
comment. Block comments and docstrings are excluded; code with a trailing
comment counts. The rule is stated here because the draft's numbers and an
earlier count of the same files disagreed by up to 30%, and the disagreement
was about what counts, not about the files.

Only the design is counted -- interface, unit bodies, topology, link rule and
fabric. Test harnesses, reference implementations and operand generators are
excluded, since the baselines are not carrying those either.

    python3 scripts/spmw_loc.py
"""

import argparse
import io
import json
import os
import sys
import tokenize


def python_loc(path):
    """Non-blank, non-comment Python lines, with docstrings excluded."""
    with open(path, "rb") as handle:
        source = handle.read()
    text = source.decode("utf-8")
    drop = set()
    try:
        tokens = list(tokenize.tokenize(io.BytesIO(source).readline))
    except tokenize.TokenError:
        tokens = []
    prev_type = None
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            drop.update(range(tok.start[0], tok.end[0] + 1))
        # A string that is a statement on its own is a docstring.
        if tok.type == tokenize.STRING and prev_type in (
            tokenize.INDENT,
            tokenize.NEWLINE,
            tokenize.NL,
            None,
        ):
            drop.update(range(tok.start[0], tok.end[0] + 1))
        if tok.type not in (tokenize.NL, tokenize.COMMENT):
            prev_type = tok.type
    count = 0
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if number in drop and not _has_code_outside_comment(line):
            continue
        count += 1
    return count


def _has_code_outside_comment(line):
    stripped = line.strip()
    return not (stripped.startswith("#") or stripped.startswith(('"""', "'''")))


def c_loc(path):
    """Non-blank, non-comment C/C++ lines, block comments excluded."""
    count = 0
    in_block = False
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if in_block:
                if "*/" in stripped:
                    in_block = False
                    rest = stripped.split("*/", 1)[1].strip()
                    if rest and not rest.startswith("//"):
                        count += 1
                continue
            if stripped.startswith("//"):
                continue
            if stripped.startswith("/*"):
                if "*/" not in stripped:
                    in_block = True
                    continue
                rest = stripped.split("*/", 1)[1].strip()
                if rest and not rest.startswith("//"):
                    count += 1
                continue
            count += 1
    return count


def loc(path):
    if path.endswith(".py"):
        return python_loc(path)
    return c_loc(path)


def region(path, start_marker, end_marker=None):
    """Count only the lines between two markers, for one design in a shared file."""
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()
    begin = next(
        (i for i, l in enumerate(lines) if start_marker in l), None
    )
    if begin is None:
        raise RuntimeError(f"marker {start_marker!r} not found in {path}")
    if end_marker is None:
        stop = len(lines)
    else:
        stop = next(
            (i for i, l in enumerate(lines[begin + 1 :], begin + 1)
             if end_marker in l),
            len(lines),
        )
    scratch = path + ".region"
    with open(scratch, "w", encoding="utf-8") as handle:
        handle.writelines(lines[begin:stop])
    try:
        return loc(scratch)
    finally:
        os.unlink(scratch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--hpfft",
        default=None,
        help="directory of a hand-written HLS FFT (HP-FFT-HLS n1024/UF32), "
        "used as the FFT row's Vitis HLS baseline instead of writing one",
    )
    args = ap.parse_args()

    def path(*parts):
        return os.path.join(args.root, *parts)

    tests = ("tests", "dataflow", "spmw")
    bl = ("examples", "spmw", "baselines")

    designs = [
        ("Systolic GEMM (16x16)", {
            "SPMW": (path(*tests, "test_spmw_gemm_int8.py"), "def gemm_int8_of", "def test_"),
            "Vitis HLS": (path(*bl, "hls", "gemm_output_stationary.cpp"), None, None),
            "SYCL/oneAPI": (path(*bl, "sycl", "gemm_output_stationary.cpp"), None, None),
        }),
        ("Multi-cache GEMM", {
            "SPMW": (path(*tests, "test_spmw_daisy.py"), "def daisy_of", "def test_"),
            "Vitis HLS": (path(*bl, "hls", "gemm_multicache.cpp"), None, None),
            "SYCL/oneAPI": (path(*bl, "sycl", "gemm_multicache.cpp"), None, None),
        }),
        ("Tiled GEMM (2-level)", {
            "SPMW": (path(*tests, "test_spmw_tiled.py"), "class MacIO", "def _operands"),
            "Vitis HLS": (path(*bl, "hls", "gemm_tiled.cpp"), None, None),
            "SYCL/oneAPI": (path(*bl, "sycl", "gemm_tiled.cpp"), None, None),
        }),
        ("FFT (spatial + folded)", {
            "SPMW": (path(*tests, "test_spmw_fft.py"), "def bfly_pair", "def _operands"),
            "SYCL/oneAPI": (path(*bl, "sycl", "fft.cpp"), None, None),
        }),
        ("Mini-TPU MXU", {
            "SPMW": (path(*tests, "test_spmw_tpu.py"), "class WsIO", "def _reference"),
            "Vitis HLS": (path(*bl, "hls", "gemm_systolic.cpp"), None, None),
            "SYCL/oneAPI": (path(*bl, "sycl", "gemm_systolic.cpp"), None, None),
        }),
        ("Attention-PV (G in {1,2,4})", {
            "SPMW": (path(*tests, "test_spmw_attention.py"), "class WsIO", "def _reference"),
            "Vitis HLS": (path(*bl, "hls", "attention_pv.cpp"), None, None),
            "SYCL/oneAPI": (path(*bl, "sycl", "attention_pv.cpp"), None, None),
        }),
    ]

    if args.hpfft:
        files = [
            os.path.join(args.hpfft, name)
            for name in ("FFT.cpp", "FFT.h")
            if os.path.exists(os.path.join(args.hpfft, name))
        ]
        if files:
            total = sum(loc(f) for f in files)
            for name, sources in designs:
                if name.startswith("FFT"):
                    sources["Vitis HLS"] = ("__external__", total, None)

    columns = ["Vitis HLS", "SYCL/oneAPI", "SPMW"]
    print(f"{'Design':30s} " + " ".join(f"{c:>12s}" for c in columns) + "   Reduction")
    out = []
    for name, sources in designs:
        row = {"design": name}
        for column in columns:
            spec = sources.get(column)
            if spec is None:
                row[column] = None
                continue
            file_path, start, end = spec
            if file_path == "__external__":
                row[column] = start  # already counted
                continue
            if not os.path.exists(file_path):
                row[column] = None
                continue
            row[column] = region(file_path, start, end) if start else loc(file_path)
        cells = []
        for column in columns:
            cells.append("-" if row[column] is None else str(row[column]))
        baselines = [row[c] for c in ("Vitis HLS", "SYCL/oneAPI") if row[c]]
        if baselines and row["SPMW"]:
            reduction = f"{max(baselines)/row['SPMW']:.1f}x"
        else:
            reduction = "-"
        row["reduction"] = reduction
        out.append(row)
        print(f"{name:30s} " + " ".join(f"{c:>12s}" for c in cells) + f"   {reduction:>9s}")

    missing = sum(
        1 for r in out for c in ("Vitis HLS", "SYCL/oneAPI") if r[c] is None
    )
    if missing:
        print(f"\n{missing} baseline cell(s) have no implementation written; "
              f"those rows cannot report a reduction.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    sys.exit(main())
