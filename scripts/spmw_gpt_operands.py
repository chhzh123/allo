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

ROWS = 128  # the sequence


def shapes_for(dim, kfile, slabs_max):
    """name -> (K, N per launch, rows, shift) for an array of `dim` with a
    `kfile`-tile weight file and room for `slabs_max` output slabs per launch.

    A launch covers as many `dim`-column output slabs as the weight file holds
    K-tiles for, and never more than the netlist's row buffer allows. FFN2's
    K=4096 spills into several passes when the file is smaller than 256 tiles,
    which the host chains through the bias.
    """

    def n_per_launch(K):
        tiles = K // dim
        slabs = max(1, min(kfile // tiles, slabs_max))
        return slabs * dim

    return {
        "proj": (1024, n_per_launch(1024), ROWS, 4),
        "ffn2": (min(4096, kfile * dim), dim, ROWS, 4),
        "score": (64, n_per_launch(64), ROWS, 6),
        "ctx": (128, n_per_launch(128), ROWS, 6),
    }


def main():
    # pylint: disable=import-outside-toplevel
    from test_spmw_gpt_stage import (
        _EXP_FUSED,
        ADD,
        NRM,
        EXP_BASE,
        EXP_SHIFT,
        GMAX,
        gpt_stage_of,
        grouped_vprog,
        lane_bias,
        LOADR,
        LOADZ,
        MAX,
        MUL,
        NB,
        NBG,
        normalise_vprog,
        pass_operands,
        PROB_BITS,
        RCP_BITS,
        row_max_vprog,
        row_sum_vprog,
        SHR,
        stage_operands,
        stage_operands_multi,
        STORE,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--kfile", type=int, default=256)
    parser.add_argument(
        "--batch",
        action="store_true",
        help="batch heads into the score and context launches and query "
        "groups into the softmax passes (the lane's GRP word)",
    )
    parser.add_argument(
        "--slabs",
        type=int,
        default=None,
        help="output slabs per launch the netlist was built for",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    DIM, KFILE = args.dim, args.kfile
    slabs = args.slabs or max(1, min(KFILE // (1024 // DIM), 4))
    shapes = shapes_for(DIM, KFILE, slabs)
    K1, _, _, _ = shapes["proj"]
    sweep = K1 // DIM
    # The netlist: `slabs * ROWS` result rows and a `sweep`-tile program per
    # row are the *buffer*; every shape below tells the device smaller or
    # equal counts. Batched attention launches deliver 8,192 rows, and the
    # board netlist is elaborated for that.
    engine = gpt_stage_of(
        DIM,
        kfile=KFILE,
        rows=ROWS,
        sweep=sweep,
        slabs=slabs,
        outs=8192 if args.batch else None,
    )
    graph = spmw.elaborate(engine)
    fams = families(graph)
    _mac, _vpu, _dim, _kfile, outs_max, _sweep, words_max, STEPS = engine.spmw_parts
    print(
        f"array {DIM}x{DIM}, file {KFILE} tiles: buffer {STEPS} steps, "
        f"{outs_max} rows, {words_max} words per launch"
    )

    for name, (K, N, rows, shift) in shapes.items():
        if args.batch and name in ("score", "ctx"):
            continue
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        X = rng.integers(-8, 8, size=(rows, K)).astype(np.int8)
        Wm = rng.integers(-4, 4, size=(K, N)).astype(np.int8)
        A, W, bias, mprog, vprog, expected = stage_operands(
            X, Wm, DIM, KFILE, shift=shift, words=0
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
        mprog_full = np.zeros((words_max + 1, DIM), dtype=np.int32)
        mprog_full[: mprog.shape[0]] = mprog
        y_full = np.zeros((max(words_max, outs), DIM), dtype=np.int32)
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
    # The three softmax passes: one query-group of `DIM` lanes, `ROWS` keys
    # down the rows, through the identity tile. Self-consistent from one
    # random group of scores, exactly as attention_head_ref computes them.
    scores = rng.integers(-128, 128, size=(ROWS, DIM)).astype(np.int8)
    i32 = np.int32
    maxes = np.maximum(scores.max(axis=0), 0).astype(i32)
    arg = np.clip(EXP_BASE + ((scores.astype(i32) - maxes) >> EXP_SHIFT), 0, 30)
    exps = (np.int32(1) << arg).astype(i32)
    sums = exps.sum(axis=0)
    recip = np.where(sums > 0, (np.int32(1) << RCP_BITS) // np.maximum(sums, 1), 0)
    probs_t = (exps * recip) >> (RCP_BITS - PROB_BITS)
    running_max = np.maximum.accumulate(np.maximum(scores.astype(i32), 0), axis=0)
    running_sum = np.cumsum(exps, axis=0)
    passes = {
        "smax": (row_max_vprog(ROWS), np.zeros((DIM, NB), i32), running_max),
        "ssum": (
            row_sum_vprog(ROWS),
            np.stack([maxes, np.zeros(DIM, i32)], 1),
            running_sum,
        ),
        "snorm": (normalise_vprog(ROWS), np.stack([maxes, sums], 1), probs_t),
    }
    for name, (vprog, bias, expected) in passes.items():
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        A, W, b, mprog, vprog = pass_operands(
            scores, DIM, KFILE, vprog, lane_bias([tuple(bias.T)], DIM), words=0
        )
        a_full = np.zeros((STEPS, DIM), dtype=np.int8)
        a_full[: A.shape[0]] = A
        mprog_full = np.zeros((words_max + 1, DIM), dtype=np.int32)
        mprog_full[: mprog.shape[0]] = mprog
        y_full = np.zeros((words_max, DIM), dtype=np.int32)
        y_full[: expected.shape[0]] = expected
        arrays = {
            "A": a_full,
            "W": W,
            "Bias": b,
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
            tensor = fam["tensor"]
            steps = fam["steps"]
            if tensor == "MProg":
                steps = mprog.shape[0]
            elif tensor == "Y":
                steps = expected.shape[0]
            elif tensor == "A":
                steps = A.shape[0]
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
                    "K": 0,
                    "N": DIM,
                    "rows": ROWS,
                    "outs": int(expected.shape[0]),
                    "families": manifest,
                },
                h,
                indent=2,
            )
        print(
            f"  {name:6s} softmax pass: {A.shape[0]} steps -> {expected.shape[0]} rows"
        )
    if args.batch:
        write_batched(args, rng, fams, STEPS, words_max)
    print("wrote", args.out)


def _emit(args, name, fams, arrays, counts, meta):
    """One operand set: the family buffers, and a manifest with the counts."""
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for fam in fams:
        buf = host_buffer(fam, arrays)
        dma = _dma_name(fam)
        with open(os.path.join(out_dir, dma + ".bin"), "wb") as handle:
            handle.write(buf)
        tensor = fam["tensor"]
        manifest.append(
            {
                "name": dma,
                "file": dma + ".bin",
                "bytes": len(buf),
                "reads": fam["reads"],
                "channels": fam["channels"],
                "steps": counts.get(tensor, fam["steps"]),
                "buffer_steps": fam["steps"],
                "tensor": tensor,
            }
        )
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as h:
        json.dump(dict(meta, shape=name, families=manifest), h, indent=2)


def write_batched(args, rng, fams, STEPS, words_max):
    """The attention shapes with heads and query groups batched per launch.

    Scores and context: as many heads as the weight file holds -- eight of
    either at 256 tiles -- one `MREP` word per head and slab, so a layer's
    scores are two launches rather than thirty-two. The softmax passes: `GMAX`
    query groups per launch, the lane restarting at each `GRP` boundary with
    that group's own maximum and sum, so a pass is sixteen launches rather
    than a hundred and twenty-eight.
    """
    # pylint: disable=import-outside-toplevel
    from test_spmw_gpt_stage import (
        _EXP,
        _EXP_FUSED,
        ADD,
        EXP_BASE,
        EXP_SHIFT,
        GMAX,
        grouped_vprog,
        lane_bias,
        LOADR,
        LOADZ,
        MAX,
        MUL,
        NB,
        NRM,
        pass_operands,
        PROB_BITS,
        RCP_BITS,
        SHR,
        stage_operands_multi,
        STORE,
    )

    DIM, KFILE = args.dim, args.kfile
    i32 = np.int32
    # -- scores^T = K . Q^T and context = P . V, `heads` heads a launch --
    for name, K, N, shift in (("score", 64, ROWS, 6), ("ctx", ROWS, 64, 6)):
        per_head = (K // DIM) * (N // DIM)
        heads = min(16, KFILE // per_head)
        pairs = []
        for _h in range(heads):
            X = rng.integers(-8, 8, size=(ROWS, K)).astype(np.int8)
            Wm = rng.integers(-4, 4, size=(K, N)).astype(np.int8)
            pairs.append((X, Wm))
        A, W, bias, mprog, vprog, expected = stage_operands_multi(
            pairs, DIM, KFILE, shift, words=0
        )
        assert A.shape[0] <= STEPS, (name, A.shape)
        outs = expected.shape[0]
        a_full = np.zeros((STEPS, DIM), dtype=np.int8)
        a_full[: A.shape[0]] = A
        mprog_full = np.zeros((words_max + 1, DIM), dtype=np.int32)
        mprog_full[: mprog.shape[0]] = mprog
        y_full = np.zeros((max(words_max, outs), DIM), dtype=np.int32)
        y_full[:outs] = expected
        arrays = {
            "A": a_full,
            "W": W,
            "Bias": bias,
            "MProg": mprog_full,
            "VProg": vprog,
            "Y": y_full,
        }
        _emit(
            args,
            name,
            fams,
            arrays,
            {"A": A.shape[0], "MProg": mprog.shape[0], "Y": outs},
            {"K": K, "N": N, "rows": ROWS, "outs": outs, "heads": heads},
        )
        print(
            f"  {name:6s} {heads} heads/launch K={K} N={N} rows={ROWS} -> outs={outs} "
            f"steps={A.shape[0]} words={mprog.shape[0] - 1}"
        )
    # -- the softmax passes, GMAX query groups a launch --
    G = GMAX
    scores = [
        rng.integers(-128, 128, size=(ROWS, DIM)).astype(np.int8) for _ in range(G)
    ]
    maxes = [np.maximum(sc.max(axis=0), 0).astype(i32) for sc in scores]
    exps = [
        (
            i32(1) << np.clip(EXP_BASE + ((sc.astype(i32) - mx) >> EXP_SHIFT), 0, 30)
        ).astype(i32)
        for sc, mx in zip(scores, maxes)
    ]
    sums = [e.sum(axis=0) for e in exps]
    recips = [
        np.where(sm > 0, (i32(1) << RCP_BITS) // np.maximum(sm, 1), 0) for sm in sums
    ]
    A = np.concatenate(scores)
    zeros = np.zeros(DIM, i32)
    passes = {
        "smax": (
            [(LOADZ, 1, 0, 0), (MAX, 0, 1, 0), (STORE, 0, 0, 0)],
            lane_bias([(zeros, zeros)] * G, DIM),
            np.concatenate(
                [
                    np.maximum.accumulate(np.maximum(sc.astype(i32), 0), axis=0)
                    for sc in scores
                ]
            ),
        ),
        "ssum": (
            _EXP_FUSED + [(ADD, 0, 1, 0), (STORE, 0, 0, 0)],
            lane_bias([(mx, zeros) for mx in maxes], DIM),
            np.concatenate([np.cumsum(e, axis=0) for e in exps]),
        ),
        "snorm": (
            _EXP_FUSED + [(NRM, 1, 0, RCP_BITS - PROB_BITS), (STORE, 1, 0, 0)],
            lane_bias(list(zip(maxes, sums)), DIM),
            np.concatenate(
                [(e * rc) >> (RCP_BITS - PROB_BITS) for e, rc in zip(exps, recips)]
            ),
        ),
    }
    for name, (prog, bias, expected) in passes.items():
        vprog = grouped_vprog(prog, ROWS, G)
        A_, W, b, mprog, vprog = pass_operands(A, DIM, KFILE, vprog, bias, words=0)
        a_full = np.zeros((STEPS, DIM), dtype=np.int8)
        a_full[: A_.shape[0]] = A_
        mprog_full = np.zeros((words_max + 1, DIM), dtype=np.int32)
        mprog_full[: mprog.shape[0]] = mprog
        outs = expected.shape[0]
        y_full = np.zeros((max(words_max, outs), DIM), dtype=np.int32)
        y_full[:outs] = expected
        arrays = {
            "A": a_full,
            "W": W,
            "Bias": b,
            "MProg": mprog_full,
            "VProg": vprog,
            "Y": y_full,
        }
        _emit(
            args,
            name,
            fams,
            arrays,
            {"A": A_.shape[0], "MProg": mprog.shape[0], "Y": outs},
            {"K": 0, "N": DIM, "rows": ROWS, "outs": outs, "groups": G},
        )
        print(
            f"  {name:6s} softmax pass, {G} groups/launch: {A_.shape[0]} steps -> {outs} rows"
        )


if __name__ == "__main__":
    main()
