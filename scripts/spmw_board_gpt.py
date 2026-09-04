# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a GPT-2 medium layer on the stage engine, one stage at a time, and time it.

A layer is eight GEMMs. On this engine each is a run of launches of one of
four shapes -- see `spmw_gpt_operands.py` -- and the host's job is to walk
them, time each shape, and check one launch of each against the answer the
reference simulator computed. Per-stage time is launches x the measured time
of that shape's launch, with the weights re-fetched from HBM inside every
launch exactly as a real walk would.

Softmax runs on the device as three passes per query-group of sixteen
lanes, through the identity tile. What is *not* on the device, and is reported
as such: GELU and the two LayerNorms. The reference runs those in float on the
FPGA; here they are named as missing rather than folded in.

Deliberately numpy-free: pyxrt is built against the system Python.

    python3 scripts/spmw_board_gpt.py DIR/spmw_kernel.xclbin DIR/args.json \\
        /scratch/$USER/gpt_operands [--device 0] [--reps 200] \\
        [--shapes proj,ffn2,score,ctx]      # the GEMM-only netlist
"""

import json
import os
import sys
import time

import pyxrt

DIM = 16
CLOCK_HZ = 250e6
SEQ, HID, FFN, HEADS, HEAD = 128, 1024, 4096, 16, 64
LAYERS = 24

#: stage -> (shape, launches per layer, MACs per layer)
STAGES = [
    ("Q projection", "proj", HID // 64, SEQ * HID * HID),
    ("K projection", "proj", HID // 64, SEQ * HID * HID),
    ("V projection", "proj", HID // 64, SEQ * HID * HID),
    ("scores K.Q^T", "score", HEADS * (SEQ // 64), HEADS * SEQ * HEAD * SEQ),
    ("softmax: row max", "smax", HEADS * (SEQ // DIM), 0),
    ("softmax: exp + sum", "ssum", HEADS * (SEQ // DIM), 0),
    ("softmax: normalise", "snorm", HEADS * (SEQ // DIM), 0),
    ("context P.V", "ctx", HEADS * (HEAD // 64), HEADS * SEQ * SEQ * HEAD),
    ("output projection", "proj", HID // 64, SEQ * HID * HID),
    ("FFN1", "proj", FFN // 64, SEQ * HID * FFN),
    ("FFN2", "ffn2", HID // 16, SEQ * FFN * HID),
]
HOST_SIDE = ["LayerNorm x2", "GELU"]


class Shape:
    def __init__(self, directory):
        with open(os.path.join(directory, "manifest.json")) as handle:
            meta = json.load(handle)
        self.name = meta["shape"]
        self.outs = meta["outs"]
        self.fams = {f["name"]: f for f in meta["families"]}
        self.data = {}
        for name, fam in self.fams.items():
            with open(os.path.join(directory, fam["file"]), "rb") as handle:
                self.data[name] = handle.read()


def main():
    xclbin_path, args_path, ops_root = sys.argv[1], sys.argv[2], sys.argv[3]
    rest = sys.argv[4:]

    def opt(flag, default):
        return rest[rest.index(flag) + 1] if flag in rest else default

    device = int(opt("--device", "0"))
    reps = int(opt("--reps", "200"))
    # Which launch shapes this netlist can run. The GEMM-only netlist has no
    # lane opcodes for the softmax passes; asking it to run one is a hang, not
    # a wrong answer, so the shapes are chosen rather than discovered.
    wanted = opt("--shapes", "proj,ffn2,score,ctx,smax,ssum,snorm").split(",")

    spec = json.load(open(args_path))
    pointers = [a for a in spec if a["pointer"]]
    scalars = [a for a in spec if not a["pointer"]]

    t0 = time.time()
    dev = pyxrt.device(device)
    uuid = dev.load_xclbin(pyxrt.xclbin(xclbin_path))
    krnl = pyxrt.kernel(dev, uuid, "spmw_kernel")
    print("device open + xclbin load: %.2f s" % (time.time() - t0))

    bos, sizes = [], []
    for index, arg in enumerate(pointers):
        nbytes = arg["channels"] * arg["steps"] * (arg["width"] // 8)
        bos.append(pyxrt.bo(dev, nbytes, pyxrt.bo.normal, krnl.group_id(index)))
        sizes.append(nbytes)
    drain = [i for i, a in enumerate(pointers) if not a["reads"]][0]

    def family(arg):
        for suffix in ("_ptr", "_steps"):
            if arg["name"].endswith(suffix):
                return arg["name"][: -len(suffix)]
        return arg["name"]

    def load(shape):
        for index, arg in enumerate(pointers):
            raw = shape.data[family(arg)]
            bos[index].write(raw if arg["reads"] else bytes(len(raw)), 0)
            bos[index].sync(
                pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                min(sizes[index], len(raw)),
                0,
            )
        counts = [shape.fams[family(a)]["steps"] for a in scalars]
        expected = shape.data[family(pointers[drain])]
        return counts, expected

    def invoke(counts, timeout_ms):
        run = krnl(*bos, *counts)
        state = run.wait(timeout_ms)
        if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise SystemExit(
                "launch did not finish in %d ms (state %s): counts %s"
                % (timeout_ms, state, counts)
            )
        return run

    def check(expected, outs):
        bos[drain].sync(
            pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, sizes[drain], 0
        )
        want = expected[: outs * DIM * 4]
        got = bytes(bos[drain].map()[: len(want)])
        return got == want

    def restarted(counts, n):
        """`n` launches on one run object; the weights are re-fetched each time
        because the kernel's own DMA does it, so this is a real walk's cost."""
        run = krnl(*bos, *counts)
        run.wait()
        t = time.time()
        for _ in range(n):
            run.start()
            run.wait()
        return (time.time() - t) / n

    shapes = {}
    for name in wanted:
        d = os.path.join(ops_root, name)
        if os.path.isdir(d):
            shapes[name] = Shape(d)

    print("\nper-shape: one launch checked, then timed")
    per_launch = {}
    for name, shape in shapes.items():
        counts, expected = load(shape)
        invoke(counts, 20000)
        ok = check(expected, shape.outs)
        secs = restarted(counts, reps)
        per_launch[name] = secs
        steps = shape.fams["feed_mac_a_in_bind"]["steps"]
        busy = steps / CLOCK_HZ
        print(
            "  %-6s counts=%-28s %s | %8.1f us/launch | %6.1f us of array work | %5.1f%% busy"
            % (
                name,
                ",".join(str(c) for c in counts),
                "matches" if ok else "MISMATCH",
                secs * 1e6,
                busy * 1e6,
                100.0 * busy / secs,
            )
        )
        if not ok:
            raise SystemExit("the device disagrees with the reference simulator")

    print("\nGPT-2 medium, one layer, per stage")
    print(
        "  %-20s %-6s %8s %12s %12s %10s"
        % ("stage", "shape", "launches", "time (ms)", "MMAC", "GMAC/s")
    )
    total = 0.0
    total_macs = 0
    for label, shape, launches, macs in STAGES:
        if shape not in per_launch:
            print("  %-20s %-6s %8s %12s" % (label, shape, "-", "no operands"))
            continue
        secs = launches * per_launch[shape]
        total += secs
        total_macs += macs
        print(
            "  %-20s %-6s %8d %12.3f %12s %10s"
            % (
                label,
                shape,
                launches,
                secs * 1e3,
                "%.1f" % (macs / 1e6) if macs else "-",
                "%.2f" % (macs / secs / 1e9) if macs else "-",
            )
        )
    print(
        "  %-20s %-6s %8s %12.3f %12.1f %10.2f"
        % (
            "GEMMs, on device",
            "",
            "",
            total * 1e3,
            total_macs / 1e6,
            total_macs / total / 1e9,
        )
    )
    for item in HOST_SIDE:
        print("  %-20s %-6s %8s %12s" % (item, "-", "-", "not on device"))
    peak = DIM * DIM * CLOCK_HZ
    print(
        "\n  => %.1f ms/layer on the accelerator, %.2f s for %d layers"
        % (total * 1e3, total * LAYERS, LAYERS)
    )
    print(
        "  => %.1f%% of the array's %.1f GMAC/s over the layer's GEMMs"
        % (100.0 * total_macs / total / peak, peak / 1e9)
    )
    print("GPT_BOARD_DONE")


if __name__ == "__main__":
    main()
