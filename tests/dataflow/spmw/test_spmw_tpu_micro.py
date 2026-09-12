# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The E3 microbenchmark: a tiled int8 GEMM whose whole epilogue is on device.

E3 measured SPMW as a TPU and `experiments/e3_tpu/gemmini/` measured Gemmini as
one, and the two were never run against the same workload.  This is that
workload:

    int8 A x int8 B -> int32 accumulate -> + bias -> arithmetic shift -> ReLU
    -> clip to int8

one ``S x S x S`` tile per launch, ``TILES`` of them back to back so a steady
interval can be read off, and *nothing* on the host.  ReLU rather than GELU
because Gemmini's `AccumulatorScale` at this scope does ReLU and nothing else,
so a transformer-shaped epilogue could not be matched.

**The order is Gemmini's, not the obvious one.**  `AccumulatorScale` applies the
activation and *then* the scale, and clips on the way out to the narrow type::

    e_act = relu(e);  e_scaled = scale_func(e_act, scale);  clip(e_scaled)

so the golden model here is ``clip8(relu(acc + bias) >> shift)``.  Writing it
the other way round still produces plausible numbers, which is exactly why it
is spelled out.

**No new hardware.**  The fabric is `gpt_stage_v1.stage_engine` -- the netlist
that ran on the U280, imported rather than re-declared -- and the epilogue is a
ten-instruction VPU program built from opcodes E3 already had.  The one thing
the ISA has no opcode for is the clip's upper bound: there is a `MAX` and no
`MIN`, so ``min(x, 127)`` is spelled ``127 - max(127 - x, 0)``: five of the
program's ten instructions, where Gemmini has a wire.  That is not a detail --
it is most of why the lane costs what it does, and this exists to price it.

Run it as a script to write the stimulus out for the Gemmini driver::

    python3 tests/dataflow/spmw/test_spmw_tpu_micro.py --size 8 --out stim_S8.txt
"""

import argparse

import numpy as np
import pytest

import allo.spmw as spmw

from gpt_stage_v1 import (
    ACCN,
    LOADB,
    LOADI,
    MAX,
    NB,
    NOP,
    NPROG,
    SHR,
    STORE,
    SUB,
    file_to_stream,
    mxu_load,
    mxu_program,
    mxu_sweep,
    stage_engine,
    vpu_header,
    vpu_word,
)

#: Tiles per launch.  Sixteen gives fifteen inter-tile gaps to take a median
#: over, which is what makes "steady interval" a measurement rather than a
#: single difference.
TILES = 16

#: The clip's upper bound, and the reason for five of the ten instructions.
INT8_MAX = 127

#: The fraction of outputs the requantisation is allowed to saturate.  A shift
#: that clips nothing leaves `clip8` untested on both sides -- the two systems
#: would agree because neither ever reached the bound, not because they clip
#: alike -- and one that clips everything measures nothing else.
CLIP_LO, CLIP_HI = 0.01, 0.20


# -- the workload, generated once and used by both systems --------------------


def choose_shift(acc):
    """The smallest shift that saturates between `CLIP_LO` and `CLIP_HI`.

    Derived from the stimulus rather than picked, so the clip is exercised at
    every size instead of being a branch that happens never to be taken.
    """
    relu = np.maximum(acc, 0)
    for shift in range(32):
        frac = float(np.mean((relu >> shift) > INT8_MAX))
        if frac <= CLIP_HI:
            if frac < CLIP_LO:
                raise ValueError(
                    f"no shift saturates between {CLIP_LO} and {CLIP_HI}: "
                    f"shift {shift} clips {frac:.4f}"
                )
            return shift
    raise ValueError("the accumulator never exceeds the int8 bound")


def golden(acc, shift):
    """`clip8(relu(acc) >> shift)` -- AccumulatorScale's order, spelled out."""
    return np.clip(np.maximum(acc, 0) >> shift, -128, INT8_MAX).astype(np.int8)


def stimulus(size, tiles=TILES, seed=0):
    """One set of operands and one golden result, for both systems.

    Full-range int8 operands: a quantised GEMM's activations and weights use
    the whole type, and a narrow stimulus would keep the accumulator small
    enough that no shift saturates.

    The bias is int8-valued because Gemmini's mesh takes its top-of-array
    partial sum at `weightType` width -- `MeshWithDelays`'s ``B_TYPE`` is
    ``Vec(meshColumns, Vec(tileColumns, weightType))``, eight bits here -- so a
    wider bias is not a thing both systems can be handed.  SPMW carries it in a
    32-bit register and would take any of them.
    """
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, size=(tiles, size, size)).astype(np.int8)
    B = rng.integers(-128, 128, size=(tiles, size, size)).astype(np.int8)
    bias = rng.integers(-128, 128, size=size).astype(np.int32)
    acc = np.einsum("tik,tkj->tij", A.astype(np.int32), B.astype(np.int32)) + bias
    shift = choose_shift(acc)
    return A, B, bias, shift, golden(acc, shift)


# -- the VPU program ----------------------------------------------------------


def micro_vprog(shift, outs, depth=NPROG, clip=True):
    """Bias, accumulate, ReLU, requantise, clip, emit -- ten instructions.

    ``r1`` is the zero the ReLU and the clip compare against.  It is never
    written, and the lane zeroes its register file once per launch, so the
    constant costs no instruction; ``127`` is not so lucky, because a register
    holding it would have to be reloaded every row -- the program body runs once
    per output and has no prologue to hoist into.

    ``clip=False`` drops the five instructions that spell the clip.  That is
    not a variant of the workload -- it computes something else, and is never
    compared against Gemmini -- but running it on the *same netlist* prices the
    clip in cycles, which is the only way to say how much of SPMW's interval
    the missing `MIN` is responsible for rather than reading it off a count.
    """
    prog = [
        (LOADB, 0, 0, 0),  # r0 = the lane's bias
        (ACCN, 0, 0, 1),  # r0 += this output's one partial sum
        (MAX, 0, 1, 0),  # ReLU, against the register that stays zero
        (SHR, 0, 0, shift),  # requantise
    ]
    if clip:
        prog += [
            (LOADI, 2, 0, INT8_MAX),
            (SUB, 2, 0, 0),  # r2 = 127 - r0
            (MAX, 2, 1, 0),  # r2 = max(127 - r0, 0)
            (LOADI, 0, 0, INT8_MAX),
            (SUB, 0, 2, 0),  # r0 = 127 - max(127 - r0, 0) = min(r0, 127)
        ]
    prog += [(STORE, 0, 0, 0)]
    words = [vpu_word(*p) for p in prog]
    if len(words) > depth:
        raise ValueError(f"{len(words)} instructions, buffer holds {depth}")
    body = words + [vpu_word(NOP)] * (depth - len(words))
    return np.array([vpu_header(outs, len(words))] + body, dtype=np.int32)


def micro_operands(size, A, B, bias, shift, kfile=None, clip=True):
    """The five input tensors for one launch of `tiles` back-to-back tiles.

    The weight file holds one tile per index, so a single `MLOAD` carries every
    tile's weights and each output row is one `MSWEEP` of count 1 against its
    own tile.  The 32-bit weight link packs four int8, so that load is
    ``size * tiles / 4`` beats a row -- a quarter of a beat per tile, against
    Gemmini's one, which its mesh overlaps with compute and this does not.
    """
    tiles = A.shape[0]
    kfile = tiles if kfile is None else kfile
    if kfile % 4:
        raise ValueError(f"the packed weight file needs a multiple of 4, got {kfile}")
    if tiles > kfile:
        raise ValueError(f"{tiles} tiles do not fit a {kfile}-entry file")
    outs = tiles * size

    activations = A.reshape(outs, size)
    weights = np.zeros((size, size, kfile), dtype=np.int8)
    for t in range(tiles):
        weights[:, :, t] = B[t]

    words = [mxu_load(size * kfile // 4)]
    for t in range(tiles):
        words += [mxu_sweep(t, 1)] * size

    biases = np.zeros((size, NB), dtype=np.int32)
    biases[:, 0] = bias
    return (
        activations,
        file_to_stream(weights),
        biases,
        mxu_program(words, size),
        micro_vprog(shift, outs, clip=clip),
    )


def micro_of(size, tiles=TILES, seed=0, clip=True):
    """The E3 stage engine, programmed for the microbenchmark.

    Returned with its operands attached so `spmw_build_array.py` drives the
    cosimulation with the same stimulus the Gemmini driver is checked against,
    and with `spmw_tokens_per_transform` set to one tile's worth of outputs so
    the testbench reports a completion cycle per tile.
    """
    A, B, bias, shift, want = stimulus(size, tiles, seed)
    outs = tiles * size
    kfile = tiles
    engine = stage_engine(size, kfile, outs=outs, sweep=1)
    names = ("A", "W", "Bias", "MProg", "VProg")
    tensors = micro_operands(size, A, B, bias, shift, kfile, clip=clip)
    engine.spmw_operands = dict(zip(names, tensors))
    engine.spmw_tokens_per_transform = size * size
    engine.spmw_micro = {
        "size": size,
        "tiles": tiles,
        "shift": int(shift),
        "clip": clip,
        "instructions": int(micro_vprog(shift, outs, clip=clip)[0] & 0xFFFF),
        "expected": want.reshape(outs, size).astype(np.int32),
    }
    return engine


# -- the stimulus as the Gemmini driver reads it ------------------------------


def dump(size, tiles=TILES, seed=0):
    """The shared stimulus as whitespace-separated integers.

    A text file rather than JSON because the other consumer is a Chisel test
    with no JSON library on its classpath, and because a comparison whose two
    halves read *the same bytes* is one fewer thing to get wrong.
    """
    A, B, bias, shift, want = stimulus(size, tiles, seed)
    flat = lambda a: " ".join(str(int(v)) for v in np.asarray(a).reshape(-1))
    return "\n".join(
        [
            f"S {size}",
            f"TILES {tiles}",
            f"SHIFT {shift}",
            f"SEED {seed}",
            "BIAS " + flat(bias),
            "A " + flat(A),
            "B " + flat(B),
            "C " + flat(want),
            "",
        ]
    )


# -- tests --------------------------------------------------------------------


def test_the_stimulus_exercises_both_bounds():
    """A clip that never fires and a ReLU that never fires prove nothing.

    Both systems clip and both apply ReLU; if the workload reached neither
    bound they would agree for a reason that has nothing to do with either.
    """
    for size in (4, 8, 16):
        A, B, bias, shift, want = stimulus(size, TILES)
        acc = np.einsum("tik,tkj->tij", A.astype(np.int32), B.astype(np.int32)) + bias
        clipped = float(np.mean((np.maximum(acc, 0) >> shift) > INT8_MAX))
        assert CLIP_LO <= clipped <= CLIP_HI, (size, shift, clipped)
        assert 0.3 < float(np.mean(acc < 0)) < 0.7, "the ReLU should fire about half"
        assert want.max() == INT8_MAX and want.min() == 0


def test_relu_precedes_the_shift_in_the_reference():
    """The two orders differ, so the reference has to commit to one.

    `AccumulatorScale` activates then scales.  For an arithmetic shift the two
    orders happen to agree, but clipping before the shift does not -- and that
    is the mistake that reads as a plausible number.
    """
    acc = np.array([[[600, -600, 40000]]], dtype=np.int32)
    assert list(golden(acc, 8).reshape(-1)) == [2, 0, 127]
    wrong = np.clip(np.maximum(acc, 0), -128, INT8_MAX) >> 8
    assert list(wrong.reshape(-1)) != [2, 0, 127]


def test_the_program_is_ten_instructions_and_reads_one_psum():
    """The lane and the array must agree on how many psums an output consumes.

    Disagreeing does not compute a wrong answer; one of them blocks forever.
    """
    prog = micro_vprog(4, outs=32)
    assert int(prog[0] >> 16) == 32 and int(prog[0] & 0xFFFF) == 10
    body = [int(w) for w in prog[1 : 1 + 10]]
    assert sum(1 for w in body if (w >> 24) & 255 == ACCN) == 1
    assert sum(1 for w in body if (w >> 24) & 255 == STORE) == 1
    assert all((w >> 24) & 255 == NOP for w in prog[1 + 10 :])


def test_the_ablation_drops_exactly_the_clip():
    """The no-clip program is the same four-instruction prefix plus STORE.

    If the ablation differed anywhere else, the cycles it saves would not be
    the clip's, and attributing them to the missing `MIN` would be wrong.
    """
    full = [int(w) for w in micro_vprog(4, 32, clip=True)[1:11]]
    bare = [int(w) for w in micro_vprog(4, 32, clip=False)[1:6]]
    assert len(full) == 10 and len(bare) == 5
    assert full[:4] == bare[:4], "the prefix through the shift must be identical"
    assert full[9] == bare[4], "both must end in the same STORE"
    assert int(micro_vprog(4, 32, clip=False)[0] & 0xFFFF) == 5


def test_the_clip_is_spelled_out_of_max_and_sub():
    """`127 - max(127 - x, 0)` is `min(x, 127)`; check it rather than trust it."""
    for x in list(range(-5, 300)) + [1 << 20]:
        assert INT8_MAX - max(INT8_MAX - x, 0) == min(x, INT8_MAX)


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_one_tile_per_launch_streamed(target):
    """Four 4x4x4 tiles back to back, bit-exact against the shared golden."""
    size, tiles = 4, 4
    A, B, bias, shift, want = stimulus(size, tiles)
    tensors = micro_operands(size, A, B, bias, shift, kfile=tiles)
    engine = stage_engine(size, tiles, outs=tiles * size, sweep=1)
    Y = np.zeros((tiles * size, size), dtype=np.int32)
    spmw.build(engine, target=target)(*tensors, Y)
    np.testing.assert_array_equal(Y, want.reshape(-1, size).astype(np.int32))


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_eight_at_the_full_tile_count(target):
    """An 8x8 array, four tiles: a second size through the same netlist."""
    size, tiles = 8, 4
    A, B, bias, shift, want = stimulus(size, tiles)
    tensors = micro_operands(size, A, B, bias, shift, kfile=tiles)
    engine = stage_engine(size, tiles, outs=tiles * size, sweep=1)
    Y = np.zeros((tiles * size, size), dtype=np.int32)
    spmw.build(engine, target=target)(*tensors, Y)
    np.testing.assert_array_equal(Y, want.reshape(-1, size).astype(np.int32))


@pytest.mark.parametrize("size", [4, 8])
def test_the_measured_configuration_matches_the_golden(size):
    """The engine the cosimulation runs must compute the shared golden result.

    This is the join that makes the comparison one. The array cosimulation
    checks the RTL against ``target="ref"`` on whatever operands the fabric
    carries -- not against this file's golden -- and the Gemmini driver checks
    itself against the golden. So if the reference and the golden disagreed at
    the measured tile count, both systems would report a pass while computing
    different things, and nothing else here would notice. The other reference
    tests run four tiles; the measurements run `TILES`, and it is `TILES` that
    has to be checked.
    """
    engine = micro_of(size)
    want = engine.spmw_micro["expected"]
    arrays = [engine.spmw_operands[n] for n in ("A", "W", "Bias", "MProg", "VProg")]
    Y = np.zeros(want.shape, dtype=np.int32)
    spmw.build(engine, target="ref")(*arrays, Y)
    np.testing.assert_array_equal(Y, want)


def test_micro_of_carries_the_same_answer_it_dumps():
    """The engine's attached operands and the Gemmini text file are one thing.

    Two generators that agree today are two generators; this asserts there is
    one, by checking the dump against what the fabric was handed.
    """
    engine = micro_of(4, tiles=4)
    text = dump(4, tiles=4)
    fields = dict(
        (line.split(" ", 1)[0], line.split(" ", 1)[1]) for line in text.splitlines()
    )
    want = np.array([int(v) for v in fields["C"].split()], dtype=np.int32)
    np.testing.assert_array_equal(engine.spmw_micro["expected"].reshape(-1), want)
    assert int(fields["SHIFT"]) == engine.spmw_micro["shift"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="write the shared stimulus out")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--tiles", type=int, default=TILES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(dump(args.size, args.tiles, args.seed))
    print(f"wrote {args.out}")
