# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a Transformer model's worth of work on the deployed array.

The array is a fixed 16x16 systolic unit with a 16-lane vector unit. What makes
a model "run" on it is not a rebuild but a tiling: every matmul in BERT or
LLaMA is cut into 16x16x16 pieces, and each piece is one invocation of the same
kernel with different pointers and different counts.

The counts are the point. `steps`, the program length and the number of outputs
are all arguments now -- in the instruction stream for the array, in the control
map for the DMA -- so a model with a different sequence length or hidden size
changes the numbers the host writes, not the bitstream. This script proves that
by running two model shapes against one `.xclbin` without reprogramming the
device between them.

Deliberately numpy-free: pyxrt is built against the system Python.

    python3 scripts/spmw_board_model.py DIR/spmw_kernel.xclbin DIR/args.json \
        [--layers 1] [--model bert-base]
"""

import json
import os
import sys
import time

import pyxrt

DIM = 16  # the array's side, which is hardware


class Model:
    """A Transformer shape, and the matmuls one of its layers performs."""

    def __init__(self, name, seq, hidden, ffn, heads, layers, gated=False):
        self.name = name
        self.seq = seq
        self.hidden = hidden
        self.ffn = ffn
        self.heads = heads
        self.layers = layers
        self.gated = gated

    def matmuls(self):
        """(label, M, K, N) for one layer, in the order a block runs them."""
        s, d, f = self.seq, self.hidden, self.ffn
        out = [
            ("q", s, d, d),
            ("k", s, d, d),
            ("v", s, d, d),
            # Attention is per head, but the arithmetic totals the same:
            # heads * (s x d/heads x s) is s x d x s.
            ("scores", s, d, s),
            ("attn.v", s, s, d),
            ("proj", s, d, d),
        ]
        out.append(("ffn1", s, d, f))
        if self.gated:
            out.append(("ffn.gate", s, d, f))
        out.append(("ffn2", s, f, d))
        return out

    def tiles(self):
        """How many 16x16x16 invocations one layer costs."""
        total = 0
        for _label, m, k, n in self.matmuls():
            total += _ceil(m, DIM) * _ceil(k, DIM) * _ceil(n, DIM)
        return total

    def macs(self):
        return sum(m * k * n for _l, m, k, n in self.matmuls())


def _ceil(value, unit):
    return (value + unit - 1) // unit


MODELS = {
    "bert-base": Model("BERT-base", 128, 768, 3072, 12, 12),
    "bert-large": Model("BERT-large", 128, 1024, 4096, 16, 24),
    # LLaMA's feed-forward is gated, so it is three matrices, not two; the
    # shape is the only thing that changes here, which is the point.
    "llama-7b": Model("LLaMA-7B", 128, 4096, 11008, 32, 32, gated=True),
    "tinyllama": Model("TinyLlama-1.1B", 128, 2048, 5632, 32, 22, gated=True),
}


def main():
    xclbin_path, args_path = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    want = "bert-base"
    budget = 20000
    if "--model" in rest:
        want = rest[rest.index("--model") + 1]
    if "--invocations" in rest:
        budget = int(rest[rest.index("--invocations") + 1])

    spec = json.load(open(args_path))
    pointers = [a for a in spec if a["pointer"]]
    scalars = [a for a in spec if not a["pointer"]]

    # The host has a U250 as well as two U280s, so the index matters.
    index = int(rest[rest.index("--device") + 1]) if "--device" in rest else 0
    t0 = time.time()
    dev = pyxrt.device(index)
    uuid = dev.load_xclbin(pyxrt.xclbin(xclbin_path))
    krnl = pyxrt.kernel(dev, uuid, "spmw_kernel")
    load_s = time.time() - t0
    print("device open + xclbin load + program: %.2f s  (paid once)" % load_s)

    # One buffer per family, sized by what the array was elaborated with. The
    # host writes `steps` per invocation; the buffer is the ceiling, not the
    # shape.
    bos, sizes = [], []
    for index, arg in enumerate(pointers):
        nbytes = arg["channels"] * arg["steps"] * (arg["width"] // 8)
        bos.append(pyxrt.bo(dev, nbytes, pyxrt.bo.normal, krnl.group_id(index)))
        sizes.append(nbytes)
        print(
            "  %-26s %5d channel(s) x %3d step(s) = %6d B"
            % (arg["name"], arg["channels"], arg["steps"], nbytes)
        )

    # Real operands, laid out by `spmw_board_operands.py` using the design's
    # own index map. Zeros would not do: a zero instruction stream says "run
    # zero steps", so the array would emit nothing and the drain would wait
    # for ever -- the kernel would simply never report done.
    ops = os.path.join(os.path.dirname(args_path), "operands")
    if len(sys.argv) > 3 and os.path.isdir(sys.argv[3]):
        ops = sys.argv[3]
    expected = None
    for index, arg in enumerate(pointers):
        path = os.path.join(ops, arg["name"] + ".bin")
        with open(path, "rb") as handle:
            raw = handle.read()
        if arg["reads"]:
            bos[index].write(raw, 0)
            bos[index].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        else:
            expected = raw
            bos[index].write(bytes(len(raw)), 0)
            bos[index].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def invoke(counts):
        run = krnl(*bos, *counts)
        run.wait()

    counts = [a["steps"] for a in scalars]
    drain = [i for i, a in enumerate(scalars) if not a["reads"]]

    def timed(counts, n):
        t0 = time.time()
        for _ in range(n):
            invoke(counts)
        return (time.time() - t0) / n

    print("\nsteady-state invocation cost (counts %s):" % counts)
    invoke(counts)  # warm the path

    drain_index = [i for i, a in enumerate(pointers) if not a["reads"]][0]
    bos[drain_index].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    got = bos[drain_index].read(len(expected), 0)
    if bytes(got) == expected:
        print("  answer matches the reference simulator, byte for byte")
    else:
        wrong = sum(1 for a, b in zip(bytes(got), expected) if a != b)
        print("  MISMATCH: %d of %d bytes differ" % (wrong, len(expected)))
    per_invocation = timed(counts, 100)
    per_invocation = timed(counts, 1000)
    print("  %8.2f us per invocation" % (per_invocation * 1e6))

    # The claim under test: a different tile is a different *number*, not a
    # different netlist. Nothing is reprogrammed between these.
    print("\nthe same bitstream, other tilings (no rebuild, no reprogram):")
    for outs in (16, 8, 4):
        alt = list(counts)
        for i in drain:
            alt[i] = outs
        print("  outs=%2d: %8.2f us" % (outs, timed(alt, 200) * 1e6))

    print()
    model = MODELS[want]
    tiles, macs = model.tiles(), model.macs()
    print(
        "%s: seq %d, hidden %d, ffn %d, %d layers"
        % (model.name, model.seq, model.hidden, model.ffn, model.layers)
    )
    print("  {:,} MAC/layer -> {:,} invocations of 16x16x16".format(macs, tiles))

    layers = int(rest[rest.index("--layers") + 1]) if "--layers" in rest else 1
    ran = min(tiles * layers, budget)
    print(
        "  running %s of them for real:"
        % ("all" if ran == tiles * layers else "{:,}".format(ran))
    )
    t0 = time.time()
    for _ in range(ran):
        invoke(counts)
    wall = time.time() - t0
    rate = wall / ran
    print(
        "    %.3f s for {:,} invocations -> %.2f us each".format(ran)
        % (wall, rate * 1e6)
    )
    print(
        "  => %.3f s/layer, %.2f s for all %d layers"
        % (tiles * rate, tiles * rate * model.layers, model.layers)
    )
    print(
        "  => %.2f GMAC/s effective, %.1f%% of the array's %.1f GMAC/s peak"
        % (
            macs / (tiles * rate) / 1e9,
            100.0 * (macs / (tiles * rate)) / (DIM * DIM * 250e6),
            DIM * DIM * 250e6 / 1e9,
        )
    )

    # And now a shape the bitstream has never seen, on the same loaded device.
    other = MODELS["llama-7b" if want.startswith("bert") else "bert-base"]
    print("\n%s on the same xclbin, nothing reloaded:" % other.name)
    otiles, omacs = other.tiles(), other.macs()
    sample = min(budget, otiles)
    t0 = time.time()
    for _ in range(sample):
        invoke(counts)
    orate = (time.time() - t0) / sample
    print("  {:,} MAC/layer -> {:,} invocations".format(omacs, otiles))
    print("  {:,} of them measured at %.2f us each".format(sample) % (orate * 1e6))
    print(
        "  => %.2f s/layer, %.0f s for all %d layers"
        % (otiles * orate, otiles * orate * other.layers, other.layers)
    )


if __name__ == "__main__":
    main()
