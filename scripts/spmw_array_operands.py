# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One launch's operands for any registry design, laid out as its feeders read them.

`spmw_board_operands.py` writes the mini-TPU's tiles; this does the same for
any design `spmw_build_array.py` knows (the array builder's own small-integer
stimulus and the reference simulator's answer), so the kernel-level simulator
(`spmw_kernel_sim.py`, behavioural AXI RAM) and the board runner can drive a
GEMM or an FFT kernel without a design-specific writer.

    python3 scripts/spmw_array_operands.py --design autosa --size 8 --out DIR

Writes DIR/<feeder>.bin per boundary family, the drain's expected bytes, and
DIR/manifest.json in the form the kernel simulator and the runner read
({"families": [...], "outs": N}).
"""

import argparse
import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "tests", "dataflow", "spmw")
)
sys.path.insert(0, os.path.dirname(__file__))

import allo.spmw as spmw  # pylint: disable=wrong-import-position
from allo.spmw.shell import (  # pylint: disable=wrong-import-position
    _dma_name,
    families,
    host_buffer,
)
from spmw_build_array import design, operands  # pylint: disable=wrong-import-position


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    fabric = design(args.design, args.size)
    graph = spmw.elaborate(fabric)
    arrays = operands(fabric, graph)
    os.makedirs(args.out, exist_ok=True)
    manifest = []
    outs = 0
    for fam in families(graph):
        buf = host_buffer(fam, arrays)
        name = _dma_name(fam)
        with open(os.path.join(args.out, name + ".bin"), "wb") as handle:
            handle.write(buf)
        manifest.append(
            {
                "name": name,
                "file": name + ".bin",
                "bytes": len(buf),
                "reads": fam["reads"],
                "channels": fam["channels"],
                "steps": fam["steps"],
                "width": fam["width"],
                "tensor": fam["tensor"],
            }
        )
        if not fam["reads"]:
            outs = fam["steps"]
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"families": manifest, "outs": outs}, handle, indent=2)
    print(f"{len(manifest)} family file(s) in {args.out}")
    for entry in manifest:
        kind = "reads" if entry["reads"] else "drain"
        print(
            f"  {entry['name']:>28} {kind:>5} steps={entry['steps']:<6d} "
            f"channels={entry['channels']:<4d} width={entry['width']:<4d} bytes={entry['bytes']}"
        )


if __name__ == "__main__":
    main()
