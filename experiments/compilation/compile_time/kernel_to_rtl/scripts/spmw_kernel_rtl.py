# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW's memory-fed kernel (scripts/spmw_package_kernel.py --design autosa --sim),
with each role's and feeder's `export_design` line removed: frontend + HLS to RTL.

The kernel reads every unit's RTL from prj/sol/syn/verilog -- the package step and
the feeders' AXI-width probe both do -- so the export is not on its path to RTL.
Everything else is the packager's own code, run through its own `main`.

    python3 spmw_kernel_rtl.py <allo tree> --size 32 --jobs 8 --out DIR
"""
import os
import sys

TREE = sys.argv.pop(1)
sys.path.insert(0, os.path.join(TREE, "scripts"))

import spmw_package_kernel as spk  # noqa: E402

EXPORT = "export_design -format ip_catalog\n"


def _strip(base, names):
    for name in names:
        path = os.path.join(base, name, "run.tcl")
        with open(path, encoding="utf-8") as handle:
            tcl = handle.read()
        assert EXPORT in tcl, path
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(tcl.replace(EXPORT, ""))
    return names


_stage, _stage_feeders = spk.stage, spk.stage_feeders
spk.stage = lambda graph, out, *a, **k: _strip(out, _stage(graph, out, *a, **k))
spk.stage_feeders = lambda graph, out, *a, **k: _strip(
    os.path.join(out, "feeders"), _stage_feeders(graph, out, *a, **k)
)
sys.argv = [
    sys.argv[0],
    "--design",
    "autosa",
    "--frequency",
    "300",
    "--sim",
] + sys.argv[1:]
spk.main()
