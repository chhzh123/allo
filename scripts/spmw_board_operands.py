# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lay out one invocation's operands the way the device's feeders read them.

The board host is numpy-free -- pyxrt is built against the system Python -- so
the buffers are prepared here, in the environment that has the design, and
written as raw bytes. `shell.host_buffer` does the ordering, which is the same
index map the reference simulator and the cosim testbench use; the device is
therefore driven by the design's own account of what each channel wants rather
than by a second guess at it.

    python3 scripts/spmw_board_operands.py --design transformer16 --size 16 \
        --out DIR/operands

Writes one file per boundary family plus `expected.bin` for the drain, so the
host can check the array's answer rather than only time it.
"""

import argparse
import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "tests", "dataflow", "spmw")
)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # pylint: disable=wrong-import-position

import allo.spmw as spmw  # pylint: disable=wrong-import-position
from allo.spmw.shell import (  # pylint: disable=wrong-import-position
    _dma_name,
    families,
    host_buffer,
)
from spmw_build_array import design  # pylint: disable=wrong-import-position


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default="transformer16")
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # pylint: disable=import-outside-toplevel
    from test_spmw_transformer import BIG, matmul_prog, requant
    from test_spmw_tpu_isa import NB, NW

    shape = BIG
    graph = spmw.elaborate(design(args.design, args.size))
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    dim, outs, steps = shape.dim, shape.outs, shape.steps
    acts = rng.integers(-8, 8, size=(steps, dim), dtype=np.int64).astype(np.int8)
    weights = rng.integers(-4, 4, size=(dim, dim, NW), dtype=np.int64).astype(np.int8)
    bias = np.zeros((dim, NB), dtype=np.int32)
    # A plain matmul and the requantise that goes with it -- the same pair the
    # block's projection steps use, so the answer is one the suite already
    # checks rather than a program invented for the board.
    mprog = matmul_prog(0, shape=shape)
    vprog = requant(4, outs=outs)

    out = np.zeros((outs, dim), dtype=np.int32)
    spmw.build(shape.engine, target="ref")(acts, weights, bias, mprog, vprog, out)

    arrays = {
        "A": acts,
        "W": weights,
        "Bias": bias,
        "MProg": mprog,
        "VProg": vprog,
        "Y": out,
    }
    manifest = []
    for fam in families(graph):
        buf = host_buffer(fam, arrays)
        name = _dma_name(fam)
        path = os.path.join(args.out, name + ".bin")
        with open(path, "wb") as handle:
            handle.write(buf)
        manifest.append(
            {
                "name": name,
                "file": os.path.basename(path),
                "bytes": len(buf),
                "reads": fam["reads"],
                "channels": fam["channels"],
                "steps": fam["steps"],
                "tensor": fam["tensor"],
            }
        )
        print(
            f"  {name:26s} {len(buf):>7d} B  {'in ' if fam['reads'] else 'out'}"
            f"  from {fam['tensor']}"
        )
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"wrote {len(manifest)} operand file(s) to {args.out}")
    drain = [f for f in families(graph) if not f["reads"]][0]
    print(
        f"the reference answer is in {_dma_name(drain)}.bin "
        f"({out.shape[0]}x{out.shape[1]} int32)"
    )


if __name__ == "__main__":
    main()
