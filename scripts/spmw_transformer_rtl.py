# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a whole Transformer block in RTL, on one netlist.

`scripts/spmw_build_array.py --design transformer --cosim` builds the engine and
simulates *one* invocation. A Transformer block is eleven, and the claim worth
checking is that they all run on the same hardware -- the same exported IPs, the
same fabric, differing only in the weights and the two programs handed to them.

So this builds nothing. It replays the trace `test_spmw_transformer.py` records
while computing the block, re-rendering the testbench for each step and running
xsim against the already-exported design::

    python3 scripts/spmw_build_array.py --design transformer --out DIR --cosim
    python3 scripts/spmw_transformer_rtl.py DIR

Each step is compared against the reference simulator's answer for that step, so
a failure names the step that broke rather than the block.
"""

import os
import subprocess
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "tests", "dataflow", "spmw")
)

import allo.spmw as spmw  # pylint: disable=wrong-import-position
from allo.spmw.cosim import render_testbench  # pylint: disable=wrong-import-position


def replay(out, verbose=True):
    """Every step of the block, one xsim run each. Returns the results."""
    # pylint: disable=import-outside-toplevel
    from test_spmw_transformer import _params, _ref_block, engine, transformer_block

    import numpy as np

    params = _params()
    block, eng = transformer_block(*params)
    if not np.array_equal(block, _ref_block(*params)):
        raise SystemExit("the reference engine disagrees with the reference block")

    graph = spmw.elaborate(engine)
    sim = os.path.join(out, "sim")
    if not os.path.isdir(sim):
        raise SystemExit(
            f"{sim} does not exist; run spmw_build_array.py --design transformer "
            f"--out {out} --cosim first"
        )

    results = []
    for index, step in enumerate(eng.trace, start=1):
        arrays = {k: v for k, v in step.items() if k != "name"}
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
                raise SystemExit(f"{command[0]} failed on step {step['name']}")
        done = subprocess.run(
            ["xsim", "sim", "-runall", "--nolog"],
            cwd=sim,
            capture_output=True,
            text=True,
            check=False,
        )
        verdict = [line for line in done.stdout.splitlines() if "SPMW COSIM" in line]
        cycles = [line for line in done.stdout.splitlines() if "SPMW CYCLES" in line]
        ok = any("PASS" in line for line in verdict)
        results.append((step["name"], ok, verdict, cycles))
        if verbose:
            print(
                f"  {index:2}. {step['name']:9} "
                + (" ".join(verdict + cycles) if verdict else "no verdict")
            )
    return results


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = sys.argv[1]
    print("replaying a Transformer block against the exported netlist:")
    results = replay(out)
    failed = [name for name, ok, _v, _c in results if not ok]
    print(
        f"{len(results) - len(failed)}/{len(results)} steps PASS"
        + (f" -- failed: {failed}" if failed else "")
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
