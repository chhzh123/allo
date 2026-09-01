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
    parser.add_argument(
        "--outs",
        type=int,
        nargs="+",
        default=[16],
        help="write one operand set per output count. The count lives in the "
        "VPU program's header and in the drain's argument, and the two have to "
        "agree: telling the drain to collect fewer rows than the lane emits "
        "leaves the array holding results nobody read, which desynchronises "
        "the next invocation. A tiling is a program, not just a number.",
    )
    args = parser.parse_args()

    # pylint: disable=import-outside-toplevel
    from test_spmw_transformer import Shape, matmul_prog, requant
    from test_spmw_tpu_isa import NB, NW

    graph = spmw.elaborate(design(args.design, args.size))
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for outs in args.outs:
        _one(graph, outs, rng, args.out, Shape, matmul_prog, requant, NB, NW)


def _one(graph, outs, rng, root, Shape, matmul_prog, requant, NB, NW):
    """One operand set: a tile that produces `outs` rows of results.

    The *reference* comes from an engine elaborated at exactly this tile --
    `Shape(16, outs)`, whose activation buffer is 2*outs rows and whose result
    is outs rows. The *device* runs the big netlist and is simply told smaller
    counts. If the two agree, the tile is data and not hardware, which is the
    whole claim; and it is checked rather than asserted.

    The layouts line up because a feeder walks its buffer step-major
    (`src[t * channels + c]`), so a smaller tile's buffer is a prefix of the
    bigger one's, with the same channel count -- and the channel count is the
    array's width, which really is hardware.
    """
    shape = Shape(16, outs)
    out_dir = os.path.join(root, f"outs{outs}")
    os.makedirs(out_dir, exist_ok=True)

    dim, steps = shape.dim, shape.steps
    acts = rng.integers(-8, 8, size=(steps, dim), dtype=np.int64).astype(np.int8)
    weights = rng.integers(-4, 4, size=(dim, dim, NW), dtype=np.int64).astype(np.int8)
    bias = np.zeros((dim, NB), dtype=np.int32)
    # A plain matmul and the requantise that goes with it -- the same pair the
    # block's projection steps use, so the answer is one the suite already
    # checks rather than a program invented for the board.
    mprog = matmul_prog(0, shape=shape)
    vprog = requant(4, outs=outs)

    result = np.zeros((outs, dim), dtype=np.int32)
    spmw.build(shape.engine, target="ref")(acts, weights, bias, mprog, vprog, result)

    arrays = {
        "A": acts,
        "W": weights,
        "Bias": bias,
        "MProg": mprog,
        "VProg": vprog,
        "Y": result,
    }
    manifest = []
    for fam in families(spmw.elaborate(shape.engine)):
        buf = host_buffer(fam, arrays)
        name = _dma_name(fam)
        with open(os.path.join(out_dir, name + ".bin"), "wb") as handle:
            handle.write(buf)
        manifest.append(
            {
                "name": name,
                "file": name + ".bin",
                "bytes": len(buf),
                "reads": fam["reads"],
                "channels": fam["channels"],
                "steps": fam["steps"],
                "tensor": fam["tensor"],
            }
        )
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    counts = {m["name"]: m["steps"] for m in manifest}
    print(f"  outs={outs:<3d} -> {len(manifest)} file(s) in {out_dir}")
    print(f"           counts {counts}")


if __name__ == "__main__":
    main()
