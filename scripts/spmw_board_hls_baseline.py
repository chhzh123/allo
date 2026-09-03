# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fig. 10's hand-written HLS baseline, on the card.

The SPMW kernel takes one 16x16x16 tile per invocation, so its board throughput
is bound by the PCIe round trip rather than by the array. This baseline can
batch many tiles into one invocation, so it is measured **both ways**: at one
tile per launch, which is the like-for-like comparison against SPMW, and
batched, which is what the design can actually do. The difference between the
two is the size of the host-interface effect, stated rather than hidden.

Deliberately numpy-free: pyxrt is built against the system Python.

    python3 scripts/spmw_board_hls_baseline.py DIR/gemm.xclbin [--device 0]
"""

import sys
import time

import pyxrt

DIM = 16
SHIFT = 6
STEPS = 16  # rows of A streamed per tile
BATCH = 1024  # tiles resident on the card at once
CLOCK_HZ = 300e6


def _s8(value):
    """Interpret a byte as a signed 8-bit integer."""
    return value - 256 if value > 127 else value


def reference(a_bytes, w_bytes, steps, tiles):
    """What the kernel should produce, in pure Python."""
    out = bytearray(steps * tiles * DIM)
    for t in range(tiles):
        w = [[_s8(w_bytes[t * DIM * DIM + r * DIM + c]) for c in range(DIM)]
             for r in range(DIM)]
        for k in range(steps):
            base = (t * steps + k) * DIM
            a = [_s8(a_bytes[base + r]) for r in range(DIM)]
            for c in range(DIM):
                acc = sum(a[r] * w[r][c] for r in range(DIM))
                acc = acc if acc > 0 else 0
                out[base + c] = (acc >> SHIFT) & 0xFF
    return bytes(out)


def main():
    xclbin_path = sys.argv[1]
    rest = sys.argv[2:]
    device = int(rest[rest.index("--device") + 1]) if "--device" in rest else 0

    t0 = time.time()
    dev = pyxrt.device(device)
    uuid = dev.load_xclbin(pyxrt.xclbin(xclbin_path))
    krnl = pyxrt.kernel(dev, uuid, "gemm_systolic")
    print("device open + xclbin load: %.2f s" % (time.time() - t0), flush=True)

    a_bytes = BATCH * STEPS * DIM
    w_bytes = BATCH * DIM * DIM
    y_bytes = BATCH * STEPS * DIM
    bo_a = pyxrt.bo(dev, a_bytes, pyxrt.bo.normal, krnl.group_id(0))
    bo_w = pyxrt.bo(dev, w_bytes, pyxrt.bo.normal, krnl.group_id(1))
    bo_y = pyxrt.bo(dev, y_bytes, pyxrt.bo.normal, krnl.group_id(2))

    a_data = bytes(((i * 7 + 3) % 15) - 7 & 0xFF for i in range(a_bytes))
    w_data = bytes(((i * 5 + 1) % 13) - 6 & 0xFF for i in range(w_bytes))
    bo_a.write(a_data, 0)
    bo_w.write(w_data, 0)
    bo_a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, a_bytes, 0)
    bo_w.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, w_bytes, 0)

    def run(tiles, timeout_ms=0):
        """One invocation. A bounded wait, because a stall here is a deadlock in
        the array rather than slowness, and an unbounded wait hides it."""
        handle = krnl(bo_a, bo_w, bo_y, STEPS, tiles)
        if timeout_ms:
            state = handle.wait(timeout_ms)
            if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise SystemExit(
                    "kernel did not finish in %d ms (state %s) at tiles=%d"
                    % (timeout_ms, state, tiles)
                )
        else:
            handle.wait()

    # Correctness on a small batch, against the pure-Python reference.
    bo_y.write(bytes(y_bytes), 0)
    bo_y.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, y_bytes, 0)
    run(2, timeout_ms=10000)
    bo_y.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, y_bytes, 0)
    got = bytes(bo_y.map()[: 2 * STEPS * DIM])
    want = reference(a_data, w_data, STEPS, 2)
    print("correctness on 2 tiles: %s" % ("matches" if got == want else "MISMATCH"), flush=True)
    if got != want:
        raise SystemExit("the device disagrees with the reference")

    def timed(tiles, reps):
        run(tiles)  # warm
        start = time.time()
        for _ in range(reps):
            run(tiles)
        return (time.time() - start) / reps

    single = timed(1, 500)
    batched = timed(BATCH, 20)
    per_tile_batched = batched / BATCH
    print("\none tile per launch : %8.2f us" % (single * 1e6))
    print("%d tiles per launch : %8.2f us  (%.3f us/tile)"
          % (BATCH, batched * 1e6, per_tile_batched * 1e6))
    print("host-interface effect: %.0fx" % (single / per_tile_batched))

    peak = DIM * DIM * CLOCK_HZ / 1e9
    print("\n%-9s %10s %12s %12s %12s" %
          ("GEMM", "tiles", "GOP/s 1-tile", "GOP/s batched", "% of peak"))
    for size in (64, 128, 256, 512, 1024):
        macs = size ** 3
        tiles = macs // (STEPS * DIM * DIM)
        for label, per_tile in (("", single), ("batched", per_tile_batched)):
            pass
        gops_single = 2.0 * macs / (tiles * single) / 1e9
        gops_batch = 2.0 * macs / (tiles * per_tile_batched) / 1e9
        pct = 100.0 * (macs / (tiles * per_tile_batched)) / (DIM * DIM * CLOCK_HZ)
        print("%-9s %10d %12.3f %12.3f %11.1f%%"
              % ("%d^3" % size, tiles, gops_single, gops_batch, pct))
    print("\narray peak: %.1f GMAC/s at %.0f MHz" % (peak, CLOCK_HZ / 1e6))
    print("HLS_BASELINE_DONE")


if __name__ == "__main__":
    main()
