# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the whole Transformer block on the U280 in ONE process.

4.1 s per step was a fresh `./top xclbin` process each time: load a 58 MB
bitstream, program the device, transfer, run, exit. That is subprocess and PCIe
configuration, not the accelerator. Here the xclbin is loaded once and the
eleven invocations reuse it, which is what any host would actually do.

Deliberately numpy-free: pyxrt is built against the system Python, and raw
bytes are all a byte-exact comparison needs.

Run under the system Python (the one pyxrt is built for)::

    python3 scripts/spmw_board_run.py DIR/top.xclbin /scratch/$USER/trace16

Measured on a U280: 0.50 s to open and program the device, then 0.117 ms
per step in steady state -- against 4.1 s per step when each invocation
was its own process. The block goes from 45 s to 2.2 ms.
"""
import os, sys, time
import pyxrt

xclbin_path, data_dir = sys.argv[1], sys.argv[2]
names = open(os.path.join(data_dir, "steps.txt")).read().split()

t0 = time.time()
dev = pyxrt.device(0)
uuid = dev.load_xclbin(pyxrt.xclbin(xclbin_path))
krnl = pyxrt.kernel(dev, uuid, "top")
load_s = time.time() - t0
print("device open + xclbin load + program: %.2f s  (paid once)" % load_s)

# Arg order is fixed by the generated host, and the sizes identify them.
SPEC = [("A", 512), ("MProg", 2048), ("VProg", 64),
        ("W", 1024), ("Bias", 128), ("Y", 1024)]
bos = [pyxrt.bo(dev, nb, pyxrt.bo.normal, krnl.group_id(i))
       for i, (_k, nb) in enumerate(SPEC)]

times, bad = [], []
for step, name in enumerate(names):
    for i, (key, nb) in enumerate(SPEC[:-1]):
        with open(os.path.join(data_dir, "s%d_%s.bin" % (step, key)), "rb") as f:
            bos[i].write(f.read(), 0)
        bos[i].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nb, 0)
    t = time.time()
    run = krnl(*bos)
    run.wait()
    dt = time.time() - t
    bos[5].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 1024, 0)
    got = bytes(bos[5].map()[:1024])  # `read` would want numpy
    with open(os.path.join(data_dir, "s%d_Y.bin" % step), "rb") as f:
        want = f.read()
    ok = got == want
    times.append(dt)
    if not ok:
        bad.append(name)
    print("  %2d. %-9s kernel %8.3f ms  %s"
          % (step + 1, name, dt * 1e3, "MATCH" if ok else "MISMATCH"))

total = sum(times)
print("\n%d/%d steps match" % (len(names) - len(bad), len(names)))
print("kernel time, whole block: %.3f ms  (mean %.3f ms/step)"
      % (total * 1e3, total / len(names) * 1e3))
print("with the one-time load:   %.2f s" % (load_s + total))
