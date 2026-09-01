# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run model-shaped work on the deployed array, and check it.

The array is a fixed 16x16 systolic unit with a 16-lane vector unit. What makes
a model "run" on it is not a rebuild but a tiling: every matmul in BERT or
LLaMA is cut into pieces the array's width, and each piece is one invocation of
the same kernel with different pointers and different counts.

The counts are the point. `steps`, the program length and the number of outputs
are arguments now -- in the instruction stream for the array, in the control map
for the DMA -- so a different sequence length or hidden size changes the numbers
the host writes, not the bitstream. `spmw_board_operands.py --outs 16 8 4`
writes one operand set per tile, each with a reference answer computed by an
engine elaborated at *that* tile; this runs them all on the one netlist and
compares. Agreement is the claim.

Deliberately numpy-free: pyxrt is built against the system Python.

    python3 scripts/spmw_board_model.py DIR/spmw_kernel.xclbin DIR/args.json \
        OPERANDS [--device 0] [--model bert-base] [--invocations N]
"""

import json
import os
import sys
import time

import pyxrt

DIM = 16  # the array's side, which really is hardware
CLOCK_HZ = 250e6


class Model:
    """A Transformer shape, and the matmuls one of its layers performs."""

    def __init__(self, name, seq, hidden, ffn, layers, gated=False):
        self.name = name
        self.seq = seq
        self.hidden = hidden
        self.ffn = ffn
        self.layers = layers
        self.gated = gated

    def matmuls(self):
        s, d, f = self.seq, self.hidden, self.ffn
        out = [
            ("q", s, d, d),
            ("k", s, d, d),
            ("v", s, d, d),
            # Attention is per head, but heads * (s x d/heads x s) totals
            # s x d x s, so the tile count does not care how it is split.
            ("scores", s, d, s),
            ("attn.v", s, s, d),
            ("proj", s, d, d),
            ("ffn1", s, d, f),
        ]
        if self.gated:
            out.append(("ffn.gate", s, d, f))
        out.append(("ffn2", s, f, d))
        return out

    def tiles(self, outs):
        """Invocations per layer when a pass produces `outs` result rows."""
        total = 0
        for _label, m, k, n in self.matmuls():
            total += _ceil(m, outs) * _ceil(k, DIM) * _ceil(n, DIM)
        return total

    def macs(self):
        return sum(m * k * n for _l, m, k, n in self.matmuls())


def _ceil(value, unit):
    return (value + unit - 1) // unit


MODELS = {
    "bert-base": Model("BERT-base", 128, 768, 3072, 12),
    "bert-large": Model("BERT-large", 128, 1024, 4096, 24),
    # LLaMA's feed-forward is gated, so three matrices rather than two. The
    # shape is the only thing that differs here, which is the point.
    "llama-7b": Model("LLaMA-7B", 128, 4096, 11008, 32, gated=True),
    "tinyllama": Model("TinyLlama-1.1B", 128, 2048, 5632, 22, gated=True),
}


class Shape:
    """One operand set: the buffers to write and the counts to pass."""

    def __init__(self, directory):
        self.dir = directory
        self.outs = int(os.path.basename(directory).replace("outs", ""))
        with open(os.path.join(directory, "manifest.json")) as handle:
            self.manifest = {m["name"]: m for m in json.load(handle)}
        self.data = {}
        for name, entry in self.manifest.items():
            with open(os.path.join(directory, entry["file"]), "rb") as handle:
                self.data[name] = handle.read()


def main():
    xclbin_path, args_path, ops_root = sys.argv[1], sys.argv[2], sys.argv[3]
    rest = sys.argv[4:]

    def opt(flag, default):
        return rest[rest.index(flag) + 1] if flag in rest else default

    want = opt("--model", "bert-base")
    budget = int(opt("--invocations", "200000"))
    device = int(opt("--device", "0"))

    spec = json.load(open(args_path))
    pointers = [a for a in spec if a["pointer"]]
    scalars = [a for a in spec if not a["pointer"]]

    t0 = time.time()
    dev = pyxrt.device(device)
    uuid = dev.load_xclbin(pyxrt.xclbin(xclbin_path))
    krnl = pyxrt.kernel(dev, uuid, "spmw_kernel")
    print(
        "device open + xclbin load + program: %.2f s  (paid once)" % (time.time() - t0)
    )

    # Buffers sized by what the array was elaborated with: the ceiling, not the
    # shape. A smaller tile writes a prefix and passes a smaller count.
    bos, sizes = [], []
    for index, arg in enumerate(pointers):
        nbytes = arg["channels"] * arg["steps"] * (arg["width"] // 8)
        bos.append(pyxrt.bo(dev, nbytes, pyxrt.bo.normal, krnl.group_id(index)))
        sizes.append(nbytes)
        print(
            "  %-26s %4d ch x %3d step = %6d B"
            % (arg["name"], arg["channels"], arg["steps"], nbytes)
        )

    drain = [i for i, a in enumerate(pointers) if not a["reads"]][0]

    def _family(arg):
        """The family an argument belongs to: its name without the suffix."""
        for suffix in ("_ptr", "_steps"):
            if arg["name"].endswith(suffix):
                return arg["name"][: -len(suffix)]
        return arg["name"]

    def load(shape):
        """Write this tile's operands; return its counts and its answer.

        The drain buffer is zeroed first, so a pass that writes nothing fails
        the comparison instead of passing on whatever the last one left.
        """
        for index, arg in enumerate(pointers):
            raw = shape.data[_family(arg)]
            bos[index].write(raw if arg["reads"] else bytes(len(raw)), 0)
            bos[index].sync(
                pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, sizes[index], 0
            )
        counts = [shape.manifest[a["name"][: -len("_steps")]]["steps"] for a in scalars]
        return counts, shape.data[_family(pointers[drain])]

    def invoke(counts, timeout_ms=0):
        """One invocation. A bounded wait, because a stall here is not slow --
        it is a drain waiting for a token the array will never send."""
        run = krnl(*bos, *counts)
        if timeout_ms:
            state = run.wait(timeout_ms)
            if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise SystemExit(
                    "kernel did not finish in %d ms (state %s); the counts and "
                    "the instruction stream disagree." % (timeout_ms, state)
                )
        else:
            run.wait()

    def check(expected):
        bos[drain].sync(
            pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, sizes[drain], 0
        )
        got = bytes(bos[drain].read(len(expected), 0))
        return got == expected

    shapes = [
        Shape(os.path.join(ops_root, d))
        for d in sorted(os.listdir(ops_root))
        if d.startswith("outs")
    ]
    shapes.sort(key=lambda s: -s.outs)
    if not shapes:
        raise SystemExit("no operand sets in " + ops_root)

    print("\nevery tile on the one netlist, nothing reprogrammed between them:")
    rates = {}
    for shape in shapes:
        counts, expected = load(shape)
        invoke(counts, timeout_ms=10000)
        ok = check(expected)
        t0 = time.time()
        for _ in range(2000):
            invoke(counts)
        rate = (time.time() - t0) / 2000
        rates[shape.outs] = rate
        print(
            "  outs=%-3d counts=%-28s %8.2f us  %s"
            % (
                shape.outs,
                ",".join(str(c) for c in counts),
                rate * 1e6,
                "matches the reference" if ok else "MISMATCH",
            )
        )
        if not ok:
            raise SystemExit("the device disagrees with the reference simulator")

    big = shapes[0]
    counts, expected = load(big)
    rate = rates[big.outs]

    print()
    for key in (want, "llama-7b" if not want.startswith("llama") else "bert-base"):
        model = MODELS[key]
        tiles = model.tiles(big.outs)
        macs = model.macs()
        print(
            "%s: seq %d, hidden %d, ffn %d, %d layers"
            % (model.name, model.seq, model.hidden, model.ffn, model.layers)
        )
        print(
            "  {:,} MAC/layer -> {:,} invocations of {}x{}x{}".format(
                macs, tiles, big.outs, DIM, DIM
            )
        )
        ran = min(tiles, budget)
        t0 = time.time()
        for _ in range(ran):
            invoke(counts)
        wall = time.time() - t0
        measured = wall / ran
        note = "the whole layer" if ran == tiles else "{:,} of them".format(ran)
        print("    ran %s in %.3f s -> %.2f us each" % (note, wall, measured * 1e6))
        print(
            "  => %.3f s/layer, %.1f s for all %d layers"
            % (tiles * measured, tiles * measured * model.layers, model.layers)
        )
        print(
            "  => %.3f GMAC/s, %.2f%% of the array's %.1f GMAC/s peak"
            % (
                macs / (tiles * measured) / 1e9,
                100.0 * macs / (tiles * measured) / (DIM * DIM * CLOCK_HZ),
                DIM * DIM * CLOCK_HZ / 1e9,
            )
        )
        print()


if __name__ == "__main__":
    main()
