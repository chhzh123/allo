# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A persistent launch server for the stage engine: one xclbin load, many jobs.

Deliberately numpy-free: pyxrt is built against the system Python. The block
orchestrator (`spmw_gpt_block.py`, numpy) writes each launch's operands as a
shape directory in the walker's format (manifest.json + one .bin per family)
and sends one line per job on stdin:

    RUN <shape_dir> <out_file>      launch once; write the drain bytes to out_file
    QUIT

The reply on stdout, one line per job:

    DONE <shape_dir> load_ms=<a> kernel_ms=<b> read_ms=<c> outs=<n>

`kernel_ms` is the time from run.start() to wait() returning (one run handle
a launch shape, restarted, as the walker's timed loop does); `load_ms` the
host->device writes and syncs; `read_ms` the drain sync and copy. Nothing is
checked here -- the orchestrator compares the bytes it gets back.

    /usr/bin/python3 scripts/spmw_xrt_runner.py XCLBIN ARGS_JSON [--device 0] [--timeout-ms 20000]
"""
import json
import os
import sys
import time

import pyxrt  # pylint: disable=import-error


def main():
    xclbin_path, args_path = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]

    def opt(flag, default):
        return rest[rest.index(flag) + 1] if flag in rest else default

    device = int(opt("--device", "0"))
    timeout_ms = int(opt("--timeout-ms", "20000"))
    spec = json.load(open(args_path, encoding="utf-8"))
    pointers = [a for a in spec if a["pointer"]]
    scalars = [a for a in spec if not a["pointer"]]

    def family(arg):
        for suffix in ("_ptr", "_steps"):
            if arg["name"].endswith(suffix):
                return arg["name"][: -len(suffix)]
        return arg["name"]

    t0 = time.time()
    dev = pyxrt.device(device)
    uuid = dev.load_xclbin(pyxrt.xclbin(xclbin_path))
    krnl = pyxrt.kernel(dev, uuid, "spmw_kernel")
    bos, sizes = [], []
    for index, arg in enumerate(pointers):
        nbytes = arg["channels"] * arg["steps"] * (arg["width"] // 8)
        bos.append(pyxrt.bo(dev, nbytes, pyxrt.bo.normal, krnl.group_id(index)))
        sizes.append(nbytes)
    drain = [i for i, a in enumerate(pointers) if not a["reads"]][0]
    runs = {}
    print("READY xclbin_load_s=%.2f" % (time.time() - t0), flush=True)

    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "QUIT":
            break
        if parts[0] != "RUN" or len(parts) < 3:
            print("ERROR bad job: " + line.strip(), flush=True)
            continue
        shape_dir, out_file = parts[1], parts[2]
        with open(os.path.join(shape_dir, "manifest.json"), encoding="utf-8") as h:
            meta = json.load(h)
        fams = {f["name"]: f for f in meta["families"]}
        data = {}
        for name, fam in fams.items():
            if fam["reads"]:
                with open(os.path.join(shape_dir, fam["file"]), "rb") as h:
                    data[name] = h.read()
        t1 = time.time()
        for index, arg in enumerate(pointers):
            fam = family(arg)
            if arg["reads"]:
                raw = data[fam]
                bos[index].write(raw, 0)
                bos[index].sync(
                    pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                    min(sizes[index], len(raw)),
                    0,
                )
        counts = [fams[family(a)]["steps"] for a in scalars]
        outs = meta["outs"]
        t2 = time.time()
        # One run handle per launch shape, restarted: what the walker's timed
        # loop does. A fresh handle a launch (krnl(...)) costs a run object
        # and its argument setup every time, 100-150 us on this host.
        run = runs.get(tuple(counts))
        if run is None:
            run = krnl(*bos, *counts)
            runs[tuple(counts)] = run
        else:
            run.start()
        state = run.wait(timeout_ms)
        t3 = time.time()
        if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            print(
                "ERROR %s launch state %s counts %s" % (shape_dir, state, counts),
                flush=True,
            )
            continue
        nbytes = outs * pointers[drain]["channels"] * (pointers[drain]["width"] // 8)
        bos[drain].sync(
            pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, sizes[drain], 0
        )
        got = bytes(bos[drain].map()[:nbytes])
        with open(out_file, "wb") as h:
            h.write(got)
        t4 = time.time()
        print(
            "DONE %s load_ms=%.3f kernel_ms=%.3f read_ms=%.3f outs=%d"
            % (shape_dir, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3, outs),
            flush=True,
        )


if __name__ == "__main__":
    main()
