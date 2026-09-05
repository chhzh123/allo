# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulate a packaged kernel against behavioural AXI RAM, with one launch's
operands, before spending hours linking it.

`spmw_package_kernel.py --sim` stages and synthesises the kernel and stops; this
takes that directory and an operand set written by `spmw_gpt_operands.py` (or
`spmw_board_operands.py`), renders `allo.spmw.kernel.kernel_testbench` with
the launch's own counts, and runs xsim. A hang here says which feeder still
has work; a mismatch says the kernel and the reference simulator disagree.

    python3 scripts/spmw_kernel_sim.py KERNEL_DIR OPERANDS_DIR --design gptstage
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "tests", "dataflow", "spmw")
)
sys.path.insert(0, os.path.dirname(__file__))

import allo.spmw as spmw  # pylint: disable=wrong-import-position
from allo.spmw.dram import ram_module  # pylint: disable=wrong-import-position
from allo.spmw.kernel import (  # pylint: disable=wrong-import-position
    arguments,
    kernel_testbench,
)
from allo.spmw.shell import _dma_name, families  # pylint: disable=wrong-import-position
from spmw_build_array import design  # pylint: disable=wrong-import-position
from spmw_package_kernel import measured_widths  # pylint: disable=wrong-import-position


def _write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _hex(path, raw):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(f"{b:02x}" for b in raw) + "\n")


def xpm_sources():
    """The XPM macros the block-RAM FIFO instantiates, from the Vivado install."""
    root = os.environ.get("XILINX_VIVADO", "")
    found = [
        os.path.join(root, "data", "ip", "xpm", sub, "hdl", sub + ".sv")
        for sub in ("xpm_memory", "xpm_fifo")
    ]
    return [f for f in found if os.path.isfile(f)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel")
    parser.add_argument("operands")
    parser.add_argument("--design", default="gptstage")
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--top", default="spmw_kernel")
    parser.add_argument("--polls", type=int, default=200000)
    args = parser.parse_args()

    graph = spmw.elaborate(design(args.design, args.size))
    fams = families(graph)
    kargs = arguments(graph)
    feeders = [_dma_name(f) for f in fams]
    widths = measured_widths(args.kernel, feeders)

    with open(os.path.join(args.operands, "manifest.json"), encoding="utf-8") as h:
        meta = json.load(h)
    by_name = {f["name"]: f for f in meta["families"]}

    sim = os.path.join(args.kernel, "sim")
    os.makedirs(sim, exist_ok=True)
    operands, counts = {}, {}
    expected = b""
    for fam in fams:
        dma = _dma_name(fam)
        entry = by_name[dma]
        with open(os.path.join(args.operands, entry["file"]), "rb") as h:
            raw = h.read()
        # The feeder reads `steps` tokens; the memory only has to hold those.
        word = fam["width"] // 8
        real = entry["steps"] * fam["channels"] * word
        counts[dma] = entry["steps"]
        if fam["reads"]:
            operands[dma] = raw[:real]
            _hex(os.path.join(sim, dma + ".hex"), raw[:real])
        else:
            operands[dma] = bytes(real)
            expected = raw[:real]
    _hex(os.path.join(sim, "expected.hex"), expected)
    print("counts:", counts)
    print("expected drain bytes:", len(expected))

    _write(
        os.path.join(sim, "tb.sv"),
        kernel_testbench(
            graph,
            kargs,
            widths,
            operands,
            expected,
            counts=counts,
            top=args.top,
            preload="hex",
            polls=args.polls,
        ),
    )
    _write(os.path.join(sim, "spmw_axi_ram.sv"), ram_module())

    sources = [os.path.join(sim, "spmw_axi_ram.sv"), os.path.join(sim, "tb.sv")]
    for name in (
        "spmw_fifo.sv",
        "spmw_const.sv",
        "spmw_top.sv",
        "spmw_control_s_axi.sv",
        f"{args.top}.sv",
    ):
        sources.append(os.path.join(args.kernel, name))
    sources += sorted(glob.glob(os.path.join(args.kernel, "roles", "*", "*_r*.sv")))
    sources += sorted(
        glob.glob(
            os.path.join(
                args.kernel, "roles", "*", "prj", "sol", "syn", "verilog", "*.v"
            )
        )
    )
    sources += sorted(
        glob.glob(
            os.path.join(
                args.kernel, "feeders", "*", "prj", "sol", "syn", "verilog", "*.v"
            )
        )
    )
    missing = [s for s in sources if not os.path.exists(s)]
    if missing:
        raise SystemExit(f"missing sources: {missing[:3]}")
    sources = xpm_sources() + sources
    print(f"{len(sources)} source file(s)")

    def run(cmd, log):
        with open(os.path.join(sim, log), "w", encoding="utf-8") as h:
            done = subprocess.run(
                cmd, cwd=sim, stdout=h, stderr=subprocess.STDOUT, check=False
            )
        return done.returncode

    t0 = time.time()
    rc = run(["xvlog", "-sv", "--incr", "--relax"] + sources, "xvlog.log")
    if rc:
        raise SystemExit(f"xvlog failed; see {sim}/xvlog.log")
    rc = run(
        [
            "xelab",
            "tb",
            "-s",
            "sim",
            "--timescale",
            "1ns/1ps",
            "--relax",
            "-debug",
            "typical",
        ],
        "xelab.log",
    )
    if rc:
        raise SystemExit(f"xelab failed; see {sim}/xelab.log")
    print(f"compiled + elaborated in {time.time() - t0:.0f}s; simulating")
    t1 = time.time()
    rc = run(["xsim", "sim", "-R"], "xsim.log")
    with open(os.path.join(sim, "xsim.log"), encoding="utf-8") as h:
        verdict = [l.strip() for l in h if "SPMW TB" in l]
    print("\n".join(verdict))
    print(f"xsim: rc={rc} in {time.time() - t1:.0f}s")
    print("KERNEL_SIM_DONE")


if __name__ == "__main__":
    main()
