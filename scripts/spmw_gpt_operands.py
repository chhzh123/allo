# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lay out one launch of each GPT-2 stage shape for the stage engine's feeders.

The board host is numpy-free, so the buffers are prepared here and written as
raw bytes in the order the device's feeders read them -- `shell.host_buffer`
does the ordering from the design's own index map. One set per *shape* of
launch, not per launch: a layer is hundreds of launches but only a few shapes,
and the host times launches and checks one of each shape against `expected`.

GPT-2 medium, sequence 128, on a 16x16 array with a 256-tile weight file:

    proj   K=1024  four 16-column slabs per launch   512 rows  64-tile sweep
    ffn2   K=4096  one slab per launch               128 rows  256-tile sweep
    score  K=64    per head, four slabs (64 keys)    512 rows  4-tile sweep
    ctx    K=128   per head, four slabs (64 dims)    512 rows  8-tile sweep

The two big shapes are 32,768 activation steps each -- the buffer the netlist
is built for -- and the two attention shapes are 2,048 and 4,096; the host
passes each launch's own counts, which is why one netlist runs a whole layer.

    python3 scripts/spmw_gpt_operands.py --out DIR/gpt_operands
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

DIM = 16
KFILE = 256
ROWS = 128  # the sequence
STEPS = 32768  # activation steps per launch, every shape

#: name -> (K, N per launch, rows, shift)
SHAPES = {
    "proj": (1024, 64, ROWS, 4),
    "ffn2": (4096, 16, ROWS, 4),
    "score": (64, 64, ROWS, 6),
    "ctx": (128, 64, ROWS, 6),
}


def main():
    # pylint: disable=import-outside-toplevel
    from test_spmw_gpt_stage import gpt_stage_of, stage_operands

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # The netlist: 512 result rows, a 64-tile sweep is the *buffer*; every
    # shape below tells the device smaller or equal counts.
    engine = gpt_stage_of(DIM)
    graph = spmw.elaborate(engine)
    fams = families(graph)

    for name, (K, N, rows, shift) in SHAPES.items():
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        X = rng.integers(-8, 8, size=(rows, K)).astype(np.int8)
        Wm = rng.integers(-4, 4, size=(K, N)).astype(np.int8)
        A, W, bias, mprog, vprog, expected = stage_operands(
            X, Wm, DIM, KFILE, shift=shift
        )
        assert A.shape[0] <= STEPS, (name, A.shape)
        outs = expected.shape[0]
        # A feeder walks its buffer step-major and reads only `steps` of it,
        # so a smaller launch's buffer is a prefix of the full one. The file is
        # laid out at the netlist's size with the real tokens in front.
        a_steps = A.shape[0]
        a_full = np.zeros((STEPS, DIM), dtype=np.int8)
        a_full[:a_steps] = A
        # Pad the program and the result to the buffer the netlist was built
        # with; the header and the count say how much of each is real.
        mprog_full = np.zeros((513, DIM), dtype=np.int32)
        mprog_full[: mprog.shape[0]] = mprog
        y_full = np.zeros((512, DIM), dtype=np.int32)
        y_full[:outs] = expected
        arrays = {
            "A": a_full,
            "W": W,
            "Bias": bias,
            "MProg": mprog_full,
            "VProg": vprog,
            "Y": y_full,
        }
        manifest = []
        for fam in fams:
            buf = host_buffer(fam, arrays)
            dma = _dma_name(fam)
            with open(os.path.join(out_dir, dma + ".bin"), "wb") as handle:
                handle.write(buf)
            # The count the host passes: real steps for this shape, which for
            # the program is words+1 and for the drain is the rows emitted.
            tensor = fam["tensor"]
            steps = fam["steps"]
            if tensor == "MProg":
                steps = mprog.shape[0]
            elif tensor == "Y":
                steps = outs
            elif tensor == "A":
                steps = a_steps
            manifest.append(
                {
                    "name": dma,
                    "file": dma + ".bin",
                    "bytes": len(buf),
                    "reads": fam["reads"],
                    "channels": fam["channels"],
                    "steps": steps,
                    "buffer_steps": fam["steps"],
                    "tensor": tensor,
                }
            )
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as h:
            json.dump(
                {
                    "shape": name,
                    "K": K,
                    "N": N,
                    "rows": rows,
                    "outs": outs,
                    "families": manifest,
                },
                h,
                indent=2,
            )
        print(
            f"  {name:6s} K={K:<5d} N/launch={N:<3d} rows={rows} -> outs={outs} "
            f"steps={A.shape[0]} words={mprog.shape[0] - 1}"
        )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
