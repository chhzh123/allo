# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Are instructions really *input*? Run programs the netlist has never seen.

`spmw_transformer_rtl.py` replays the eleven steps of a Transformer block on one
built design, which shows those eleven need no rebuild. It does not show the
general claim, because those eleven are the programs the design was written
for.

This runs programs it was not. None of them appear anywhere in the Transformer;
one of them uses an MXU opcode the block never issues at all. Nothing is
rebuilt -- the same exported IPs, the same fabric, the same `sim` directory::

    python3 scripts/spmw_build_array.py --design transformer --out DIR --cosim
    python3 scripts/spmw_program_check.py DIR

What *is* fixed in the hardware is the envelope, not the program: at most
``vprog_len`` VPU instructions, ``steps`` MXU steps feeding ``outs`` outputs,
``NW`` weight matrices per cell, ``REGS`` registers per lane, and the opcode set
itself. Anything inside that is data. `test_spmw_transformer.py` states the same
bounds as assertions.
"""

import os
import subprocess
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "tests", "dataflow", "spmw")
)

import allo.spmw as spmw  # pylint: disable=wrong-import-position
from allo.spmw.cosim import render_testbench  # pylint: disable=wrong-import-position


def _run_one(sim, graph, arrays, label):
    """Render a testbench for one program and simulate it."""
    with open(os.path.join(sim, "tb.sv"), "w", encoding="utf-8") as handle:
        handle.write(render_testbench(graph, arrays, arrays))
    for command in (
        ["xvlog", "-sv", "--nolog", "tb.sv"],
        ["xelab", "-relax", "--nolog", "-s", "sim", "tb"],
    ):
        done = subprocess.run(
            command, cwd=sim, capture_output=True, text=True, check=False
        )
        if done.returncode:
            print(done.stdout[-3000:])
            raise SystemExit(f"{command[0]} failed on {label}")
    done = subprocess.run(
        ["xsim", "sim", "-runall", "--nolog"],
        cwd=sim,
        capture_output=True,
        text=True,
        check=False,
    )
    verdict = [line for line in done.stdout.splitlines() if "SPMW COSIM" in line]
    return any("PASS" in line for line in verdict), verdict


def check(out, verbose=True):
    """Every novel program, one xsim run each."""
    # pylint: disable=import-outside-toplevel
    from test_spmw_transformer import NOVEL_PROGRAMS, engine, novel_operands

    graph = spmw.elaborate(engine)
    sim = os.path.join(out, "sim")
    if not os.path.isdir(sim):
        raise SystemExit(
            f"{sim} does not exist; run spmw_build_array.py --design transformer "
            f"--out {out} --cosim first"
        )

    results = []
    for index, (name, mprog, vprog, _fn) in enumerate(NOVEL_PROGRAMS, start=1):
        arrays = novel_operands(name)
        ok, verdict = _run_one(sim, graph, arrays, name)
        results.append((name, ok))
        if verbose:
            print(f"  {index}. {name:14} " + (" ".join(verdict) or "no verdict"))
        del mprog, vprog
    return results


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print("running programs the built design has never seen:")
    results = check(sys.argv[1])
    failed = [name for name, ok in results if not ok]
    print(
        f"{len(results) - len(failed)}/{len(results)} novel programs PASS"
        + (f" -- failed: {failed}" if failed else "")
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
