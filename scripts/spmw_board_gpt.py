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
        [--clock-mhz 300]                  # the clock the link closed at
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

#: stage -> (shape, output features, reduction length, MACs per layer).
#: Launches per layer come from the operand manifests: a launch covers
#: `N` output features and `K` of the reduction, so a stage needs
#: (features / N) * (reduction / K) of them -- which is how the same walk
#: measures a 256-tile file (FFN2 in one pass) and a 64-tile one (four).
STAGES = [
    ("Q projection", "proj", HID, HID, SEQ * HID * HID),
    ("K projection", "proj", HID, HID, SEQ * HID * HID),
    ("V projection", "proj", HID, HID, SEQ * HID * HID),
    ("scores K.Q^T", "score", SEQ * HEADS, HEAD, HEADS * SEQ * HEAD * SEQ),
    ("softmax: row max", "smax", HEADS * SEQ, 0, 0),
    ("softmax: exp + sum", "ssum", HEADS * SEQ, 0, 0),
    ("softmax: normalise", "snorm", HEADS * SEQ, 0, 0),
    ("context P.V", "ctx", HEAD * HEADS, SEQ, HEADS * SEQ * SEQ * HEAD),
    ("output projection", "proj", HID, HID, SEQ * HID * HID),
    ("FFN1", "proj", FFN, HID, SEQ * HID * FFN),
    ("FFN2", "ffn2", HID, FFN, SEQ * FFN * HID),
]
HOST_SIDE = ["LayerNorm x2", "GELU"]


class Shape:
    def __init__(self, directory):
        with open(os.path.join(directory, "manifest.json")) as handle:
            meta = json.load(handle)
        self.name = meta["shape"]
        self.outs = meta["outs"]
        self.n = meta.get("N", DIM)  # output features one launch covers
        self.k = meta.get("K", 0)  # reduction length one launch covers
        self.heads = meta.get("heads", 1)  # heads batched into one launch
        self.groups = meta.get("groups", 1)  # query groups batched into one launch
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
    # The kernel clock the bitstream was linked at: the busy figure and the
    # peak are cycles, and a faster link makes both of them worth more.
    global CLOCK_HZ  # pylint: disable=global-statement
    CLOCK_HZ = float(opt("--clock-mhz", str(CLOCK_HZ / 1e6))) * 1e6
    print(f"kernel clock taken as {CLOCK_HZ / 1e6:.0f} MHz")
    dump = int(opt("--dump", "0"))  # mismatching values to print, if any
    settle = float(opt("--settle-ms", "0")) / 1e3  # wait after done before reading
    keep_going = "--keep-going" in rest
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
        if got != want and dump:
            # Which rows and lanes disagree: a tail of stale rows is a
            # completion race, a scatter is a data path, one lane is a lane.
            rows = [
                r
                for r in range(outs)
                if got[r * DIM * 4 : (r + 1) * DIM * 4]
                != want[r * DIM * 4 : (r + 1) * DIM * 4]
            ]
            print(f"    {len(rows)} of {outs} rows differ; first {rows[:6]}, last {rows[-3:]}")
            shown = 0
            for r in rows:
                for lane in range(DIM):
                    o = (r * DIM + lane) * 4
                    g = int.from_bytes(got[o : o + 4], "little", signed=True)
                    w = int.from_bytes(want[o : o + 4], "little", signed=True)
                    if g != w and shown < dump:
                        print(f"    row {r} lane {lane}: got {g} want {w}")
                        shown += 1
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
        if settle:
            time.sleep(settle)
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
        if not ok and not keep_going:
            raise SystemExit("the device disagrees with the reference simulator")

    print("\nGPT-2 medium, one layer, per stage")
    print(
        "  %-20s %-6s %8s %12s %12s %10s"
        % ("stage", "shape", "launches", "time (ms)", "MMAC", "GMAC/s")
    )
    total = 0.0
    total_macs = 0
    for label, shape, features, reduction, macs in STAGES:
        if shape not in per_launch:
            print("  %-20s %-6s %8s %12s" % (label, shape, "-", "no operands"))
            continue
        sh = shapes[shape]
        if shape in ("smax", "ssum", "snorm"):
            # one query group of DIM lanes per launch, or `groups` of them
            launches = (features // DIM) // sh.groups
        else:
            launches = (features // sh.n) * max(1, reduction // max(sh.k, 1))
            launches //= sh.heads
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
