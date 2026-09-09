# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One GPT-2 medium decoder block, chained launch by launch on the stage engine.

Hybrid by construction, and labelled so: the matrix stages and the three
softmax passes run on the device (the frozen v1 engine `gpt_stage_v1`, the
`gptkern_302` bitstream); LayerNorm, the causal mask, GELU, the residual adds
and every int32 -> int8 requantisation run on the host in numpy. Every device
launch is checked against the integer reference the tests define the moment
its wait() returns, and each stage is fed from the previous stage's *device*
output. The block's output is checked against a numpy run of the same frozen
semantics with no device at all (`--no-device` produces that run).

The frozen semantics (synthetic weights, seed `--seed`):

    x0        int8 [128, 1024] ~ U[-8, 8)
    weights   int8 ~ U[-4, 4): Wq Wk Wv Wo [1024, 1024], W1 [1024, 4096], W2 [4096, 1024]
    LN        float32 over the hidden axis, eps 1e-5, gamma 1, beta 0, then
              clip(rint(y * 32)) -> int8
    proj      clip8((X @ W) >> 4)                     (device: `proj`, 16 launches a matrix)
    scores    clip8((K_h @ Q_h^T) >> QUANT_SCORE)     (device: `score`, 2 launches a head)
    mask      causal: score[key, query] = -128 for key > query   (host)
    softmax   the engine's three passes per (head, 16-query group): row max
              (floored at 0), exp2(clamp(EXP_BASE + ((s - max) >> EXP_SHIFT), 0, 30)),
              sum, then (exp * ((1 << RCP_BITS) // sum)) >> (RCP_BITS - PROB_BITS),
              clip8                                   (device: 3 x 128 launches)
    context   clip8((P_h @ V_h) >> PROB_BITS)         (device: `ctx`, 1 launch a head)
    out proj  clip8((attn @ Wo) >> 4); r1 = clip8(x0 + o)
    FFN1      clip8((LN(r1) @ W1) >> 4) -> GELU (tanh form) on v = q / 16, back as
              clip(rint(g * 16)) -> int8              (device: `proj`, 64 launches)
    FFN2      clip8((g @ W2) >> 4)                    (device: `ffn2`, 64 launches)
    output    clip8(r1 + f2)                          int8 [128, 1024]

    /scratch/hc676/allo-agent/bin/python3 scripts/spmw_gpt_block.py \\
        --xclbin gptkern_302/spmw_kernel.xclbin --args gptkern_302/args.json \\
        --out DIR [--seed 0] [--reps 3] [--device 0] [--no-device]
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "dataflow", "spmw"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # pylint: disable=wrong-import-position
import allo.spmw as spmw  # pylint: disable=wrong-import-position
from allo.spmw.shell import _dma_name, families, host_buffer  # pylint: disable=wrong-import-position
from gpt_stage_v1 import (  # pylint: disable=wrong-import-position
    EXP_BASE,
    EXP_SHIFT,
    NB,
    PROB_BITS,
    QUANT_SCORE,
    RCP_BITS,
    gpt_stage_of,
    normalise_vprog,
    pass_operands,
    row_max_vprog,
    row_sum_vprog,
    stage_operands,
)

MODELS = {
    # seq, hidden, heads, head width, FFN width, norm, position, activation
    "gpt2-medium": dict(seq=128, hid=1024, heads=16, head=64, ffn=4096, norm="layernorm", rope=False, gate=False),
    "llama-7b": dict(seq=128, hid=4096, heads=32, head=128, ffn=11008, norm="rmsnorm", rope=True, gate=True),
}
SEQ, HID, HEADS, HEAD, FFN = 128, 1024, 16, 64, 4096
NORM, ROPE, GATE = "layernorm", False, False
CHECK_PACKER = [False]
K_LAUNCH = 1024  # the K one launch of `proj` covers: 64 tiles of 16 in the 256-tile file with 4 slabs
DIM, KFILE, SWEEP, SLABS = 16, 256, 64, 4
PROJ_SHIFT = 4
LN_SCALE, GELU_IN, GELU_OUT = 32.0, 16.0, 16.0
I32 = np.int32

_PLANS = {}


def fast_buffer(fam, arrays):
    """`host_buffer` with the family's index map applied by numpy at once.

    The plan -- for each channel, the tensor index of each step's token -- is
    fixed per family, so its integer components become index arrays once and
    the tokens are gathered with one fancy index; a slice component (a vector
    token such as a weight word or the bias pair) is the same for every token
    and is applied as ordinary indexing after it. The bytes are those
    `host_buffer` builds token by token (`--check-packer` compares them).
    """
    key = fam["name"]
    if key not in _PLANS:
        plan = fam["plan"]
        first = plan[0][0]
        ints = [r for r, c in enumerate(first) if not isinstance(c, slice)]
        slices = [(r, c) for r, c in enumerate(first) if isinstance(c, slice)]
        for chan in plan:
            for index in chan:
                for r, c in slices:
                    if index[r] != c:
                        raise ValueError(f"`{key}`: slice component {r} varies between tokens")
        gathered = [
            np.array([[index[r] for index in chan] for chan in plan], dtype=np.int64)
            for r in ints
        ]
        _PLANS[key] = (ints, slices, gathered, len(first))
    ints, slices, gathered, rank = _PLANS[key]
    index = [None] * rank
    for r, g in zip(ints, gathered):
        index[r] = g
    for r, c in slices:
        index[r] = c
    array = arrays[fam["tensor"]]
    tokens = array[tuple(index)]  # [channels, steps, *token]
    tokens = np.ascontiguousarray(np.swapaxes(tokens, 0, 1))  # [steps, channels, *token]
    word = fam["width"] // 8
    raw = tokens.tobytes()
    if len(raw) != fam["channels"] * fam["steps"] * word:
        raise ValueError(
            f"`{fam['name']}`: packed {len(raw)} bytes, expected "
            f"{fam['channels'] * fam['steps'] * word}"
        )
    return raw


def clip8(x):
    return np.clip(x, -128, 127).astype(np.int8)


def ln_quant(x):
    xf = x.astype(np.float32)
    if NORM == "rmsnorm":
        # RMSNorm: x / sqrt(mean(x^2) + eps), weight 1
        y = xf / np.sqrt((xf * xf).mean(axis=1, keepdims=True) + np.float32(1e-5))
    else:
        mu = xf.mean(axis=1, keepdims=True)
        var = ((xf - mu) ** 2).mean(axis=1, keepdims=True)
        y = (xf - mu) / np.sqrt(var + np.float32(1e-5))
    return clip8(np.rint(y * np.float32(LN_SCALE)))


def rope_quant(x):
    """Rotary position embedding on int8 [seq, heads*head], base 10000, pairs
    (2i, 2i+1) within a head, in float32, requantised with the same scale."""
    seq, width = x.shape
    xf = x.astype(np.float32).reshape(seq, HEADS, HEAD)
    half = HEAD // 2
    inv = np.float32(10000.0) ** (-np.arange(0, half, dtype=np.float32) / np.float32(half))
    ang = np.arange(seq, dtype=np.float32)[:, None] * inv[None, :]  # [seq, half]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = xf[:, :, 0::2], xf[:, :, 1::2]
    out = np.empty_like(xf)
    out[:, :, 0::2] = x1 * cos - x2 * sin
    out[:, :, 1::2] = x1 * sin + x2 * cos
    return clip8(np.rint(out.reshape(seq, width)))


def silu_gate_quant(f, g):
    """SiLU(f / 16) * (g / 16), requantised by 16: the gated FFN's inner product."""
    v = f.astype(np.float32) / np.float32(GELU_IN)
    u = g.astype(np.float32) / np.float32(GELU_IN)
    return clip8(np.rint(v / (1.0 + np.exp(-v)) * u * np.float32(GELU_OUT)))


def gelu_quant(x):
    v = x.astype(np.float32) / np.float32(GELU_IN)
    g = 0.5 * v * (1.0 + np.tanh(np.float32(0.7978845608) * (v + 0.044715 * v**3)))
    return clip8(np.rint(g * np.float32(GELU_OUT)))


class Device:
    """The runner process (numpy-free, system Python, pyxrt) behind a pipe."""

    def __init__(self, xclbin, args, device, runner):
        self.proc = subprocess.Popen(
            ["/usr/bin/python3", runner, xclbin, args, "--device", str(device)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        line = self.proc.stdout.readline().strip()
        if not line.startswith("READY"):
            raise SystemExit(f"runner did not start: {line!r}")
        self.ready = line

    def run(self, shape_dir, out_file, outs):
        self.proc.stdin.write(f"RUN {shape_dir} {out_file}\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().strip()
        if not line.startswith("DONE"):
            raise SystemExit(f"runner: {line}")
        stats = {k: float(v) for k, v in (kv.split("=") for kv in line.split()[2:])}
        got = np.fromfile(out_file, dtype=np.int32).reshape(outs, DIM)
        return got, stats

    def close(self):
        self.proc.stdin.write("QUIT\n")
        self.proc.stdin.flush()
        self.proc.wait(timeout=30)


class Block:
    def __init__(self, out, device, seed):
        self.out = out
        self.device = device
        self.rng = np.random.default_rng(seed)
        engine = gpt_stage_of(DIM, kfile=KFILE, rows=SEQ, sweep=SWEEP, slabs=SLABS)
        graph = spmw.elaborate(engine)
        self.fams = families(graph)
        (_m, _v, _d, _k, _outs, _sweep, self.words_max, self.steps) = engine.spmw_parts
        self.launch_no = 0
        self.stats = {}
        self.mismatch = {}

    # -- one launch -----------------------------------------------------------

    def emit(self, name, A, W, bias, mprog, vprog, expected):
        outs = expected.shape[0]
        a_full = np.zeros((self.steps, DIM), dtype=np.int8)
        a_full[: A.shape[0]] = A
        mprog_full = np.zeros((self.words_max + 1, DIM), dtype=np.int32)
        mprog_full[: mprog.shape[0]] = mprog
        y_full = np.zeros((max(self.words_max, outs), DIM), dtype=np.int32)
        y_full[:outs] = expected
        arrays = {"A": a_full, "W": W, "Bias": bias, "MProg": mprog_full, "VProg": vprog, "Y": y_full}
        d = os.path.join(self.out, "launches", f"{self.launch_no:04d}_{name}")
        self.launch_no += 1
        os.makedirs(d, exist_ok=True)
        manifest = []
        for fam in self.fams:
            buf = fast_buffer(fam, arrays)
            if CHECK_PACKER[0]:
                assert buf == host_buffer(fam, arrays), fam["name"]
            dma = _dma_name(fam)
            with open(os.path.join(d, dma + ".bin"), "wb") as h:
                h.write(buf)
            tensor = fam["tensor"]
            steps = fam["steps"]
            if tensor == "MProg":
                steps = mprog.shape[0]
            elif tensor == "Y":
                steps = outs
            elif tensor == "A":
                steps = A.shape[0]
            manifest.append(
                {"name": dma, "file": dma + ".bin", "bytes": len(buf), "reads": fam["reads"],
                 "channels": fam["channels"], "steps": steps, "buffer_steps": fam["steps"],
                 "tensor": tensor}
            )
        with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as h:
            json.dump({"shape": name, "outs": outs, "families": manifest}, h)
        return d, outs

    def launch(self, stage, name, A, W, bias, mprog, vprog, expected):
        t0 = time.time()
        d, outs = self.emit(name, A, W, bias, mprog, vprog, expected)
        pack = time.time() - t0
        st = self.stats.setdefault(stage, {"launches": 0, "pack_s": 0.0, "load_s": 0.0, "kernel_s": 0.0, "read_s": 0.0, "host_s": 0.0})
        st["launches"] += 1
        st["pack_s"] += pack
        if self.device is None:
            return expected
        got, s = self.device.run(d, os.path.join(d, "drain.bin"), outs)
        st["load_s"] += s["load_ms"] / 1e3
        st["kernel_s"] += s["kernel_ms"] / 1e3
        st["read_s"] += s["read_ms"] / 1e3
        bad = int((got != expected).sum())
        if bad:
            self.mismatch[stage] = self.mismatch.get(stage, 0) + bad
            print(f"  MISMATCH {stage} {name}: {bad} of {expected.size} values", flush=True)
        return got

    # -- the stages ------------------------------------------------------------

    def proj(self, stage, X, Wm):
        """clip8(sum over K chunks of ((X_k @ Wm_k) >> PROJ_SHIFT)), N in 64-column
        launches (4 slabs of 16); a K wider than one launch (LLaMA's 4,096) is
        several launches whose shifted partial sums the host adds -- the
        frozen semantics for wide K, which is what the device computes."""
        R, K = X.shape
        N = Wm.shape[1]
        per = SLABS * DIM
        out = np.zeros((R, N), dtype=np.int32)
        for k0 in range(0, K, K_LAUNCH):
            Xk, Wk = X[:, k0 : k0 + K_LAUNCH], Wm[k0 : k0 + K_LAUNCH]
            for n0 in range(0, N, per):
                A, W, bias, mprog, vprog, expected = stage_operands(Xk, Wk[:, n0 : n0 + per], DIM, KFILE, shift=PROJ_SHIFT)
                got = self.launch(stage, f"{stage}_k{k0}_n{n0}", A, W, bias, mprog, vprog, expected)
                for s in range(SLABS):
                    out[:, n0 + s * DIM : n0 + (s + 1) * DIM] += got[s * R : (s + 1) * R]
        return clip8(out)

    def ffn2(self, stage, X, Wm):
        """Up to K = 4096 in one launch of one 16-column slab (the 256-tile
        file); a wider K (LLaMA's 11,008) is chunks of 4,096 whose shifted
        partial sums the host adds."""
        R, K = X.shape
        N = Wm.shape[1]
        out = np.zeros((R, N), dtype=np.int32)
        kc = KFILE * DIM
        for k0 in range(0, K, kc):
            Xk, Wk = X[:, k0 : k0 + kc], Wm[k0 : k0 + kc]
            for n0 in range(0, N, DIM):
                A, W, bias, mprog, vprog, expected = stage_operands(Xk, Wk[:, n0 : n0 + DIM], DIM, KFILE, shift=PROJ_SHIFT)
                got = self.launch(stage, f"{stage}_k{k0}_n{n0}", A, W, bias, mprog, vprog, expected)
                out[:, n0 : n0 + DIM] += got[:R]
        return clip8(out)

    def scores(self, Kh, Qh):
        """scores^T [key, query] = clip8((K_h @ Q_h^T) >> QUANT_SCORE), 64 queries a launch."""
        S = np.zeros((SEQ, SEQ), dtype=np.int32)
        per = SLABS * DIM
        for q0 in range(0, SEQ, per):
            A, W, bias, mprog, vprog, expected = stage_operands(Kh, Qh.T[:, q0 : q0 + per], DIM, KFILE, shift=QUANT_SCORE)
            got = self.launch("score", f"score_q{q0}", A, W, bias, mprog, vprog, expected)
            for s in range(SLABS):
                S[:, q0 + s * DIM : q0 + (s + 1) * DIM] = got[s * SEQ : (s + 1) * SEQ]
        return clip8(S)

    def softmax_group(self, rows_in):
        """One 16-query group: three passes; returns probs_t [key, query] int32."""
        s32 = rows_in.astype(I32)
        maxes_ref = np.maximum(s32.max(axis=0), 0).astype(I32)
        running_max = np.maximum.accumulate(np.maximum(s32, 0), axis=0)
        zeros = np.zeros((DIM, NB), dtype=I32)
        A, W, b, mprog, vprog = pass_operands(rows_in, DIM, KFILE, row_max_vprog(SEQ), zeros)
        got = self.launch("smax", "smax", A, W, b, mprog, vprog, running_max)
        maxes = got[SEQ - 1].astype(I32)
        arg = np.clip(EXP_BASE + ((s32 - maxes) >> EXP_SHIFT), 0, 30)
        exps = (I32(1) << arg).astype(I32)
        running_sum = np.cumsum(exps, axis=0).astype(I32)
        bias = np.stack([maxes, np.zeros(DIM, I32)], 1)
        A, W, b, mprog, vprog = pass_operands(rows_in, DIM, KFILE, row_sum_vprog(SEQ), bias)
        got = self.launch("ssum", "ssum", A, W, b, mprog, vprog, running_sum)
        sums = got[SEQ - 1].astype(I32)
        recip = np.where(sums > 0, (I32(1) << RCP_BITS) // np.maximum(sums, 1), 0).astype(I32)
        probs_t = ((exps * recip) >> (RCP_BITS - PROB_BITS)).astype(I32)
        bias = np.stack([maxes, sums], 1)
        A, W, b, mprog, vprog = pass_operands(rows_in, DIM, KFILE, normalise_vprog(SEQ), bias)
        got = self.launch("snorm", "snorm", A, W, b, mprog, vprog, probs_t)
        assert (maxes == maxes_ref).all() or self.device is not None
        return got

    def context(self, probs, Vh):
        """clip8((P_h @ V_h) >> PROB_BITS), the head's width in 64-column launches."""
        per = SLABS * DIM
        out = np.zeros((SEQ, HEAD), dtype=np.int32)
        for n0 in range(0, HEAD, per):
            A, W, bias, mprog, vprog, expected = stage_operands(probs, Vh[:, n0 : n0 + per], DIM, KFILE, shift=PROB_BITS)
            got = self.launch("ctx", f"ctx_n{n0}", A, W, bias, mprog, vprog, expected)
            for s in range(SLABS):
                out[:, n0 + s * DIM : n0 + (s + 1) * DIM] = got[s * SEQ : (s + 1) * SEQ]
        return clip8(out)

    def run(self):
        rng = self.rng
        x0 = rng.integers(-8, 8, (SEQ, HID)).astype(np.int8)
        Wq, Wk, Wv, Wo = (rng.integers(-4, 4, (HID, HID)).astype(np.int8) for _ in range(4))
        W1 = rng.integers(-4, 4, (HID, FFN)).astype(np.int8)
        W2 = rng.integers(-4, 4, (FFN, HID)).astype(np.int8)
        t0 = time.time()
        W3 = rng.integers(-4, 4, (HID, FFN)).astype(np.int8) if GATE else None
        h = self.host("ln1", lambda: ln_quant(x0))
        Q = self.proj("Q", h, Wq)
        K = self.proj("K", h, Wk)
        V = self.proj("V", h, Wv)
        if ROPE:
            Q = self.host("rope", lambda: rope_quant(Q))
            K = self.host("rope", lambda: rope_quant(K))
        attn = np.zeros((SEQ, HID), dtype=np.int8)
        for hd in range(HEADS):
            sl = slice(hd * HEAD, (hd + 1) * HEAD)
            S = self.scores(K[:, sl], Q[:, sl])  # [key, query]
            S = self.host("mask", lambda S=S: np.where(np.arange(SEQ)[:, None] > np.arange(SEQ)[None, :], np.int8(-128), S))
            probs = np.zeros((SEQ, SEQ), dtype=np.int8)  # [query, key]
            for g in range(SEQ // DIM):
                pt = self.softmax_group(S[:, g * DIM : (g + 1) * DIM])
                probs[g * DIM : (g + 1) * DIM] = clip8(pt.T)
            attn[:, sl] = self.context(probs, V[:, sl])
        o = self.proj("O", attn, Wo)
        r1 = self.host("res1", lambda: clip8(x0.astype(I32) + o.astype(I32)))
        h2 = self.host("ln2", lambda: ln_quant(r1))
        f1 = self.proj("FFN1", h2, W1)
        if GATE:
            f3 = self.proj("FFN3", h2, W3)
            g = self.host("silu_gate", lambda: silu_gate_quant(f1, f3))
        else:
            g = self.host("gelu", lambda: gelu_quant(f1))
        f2 = self.ffn2("FFN2", g, W2)
        out = self.host("res2", lambda: clip8(r1.astype(I32) + f2.astype(I32)))
        self.wall = time.time() - t0
        return out

    def host(self, stage, fn):
        t0 = time.time()
        r = fn()
        st = self.stats.setdefault(stage, {"launches": 0, "pack_s": 0.0, "load_s": 0.0, "kernel_s": 0.0, "read_s": 0.0, "host_s": 0.0})
        st["host_s"] += time.time() - t0
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin")
    ap.add_argument("--args")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--no-device", action="store_true", help="the numpy reference run: every launch's expected output stands in for the device")
    ap.add_argument("--runner", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "spmw_xrt_runner.py"))
    ap.add_argument("--check-packer", action="store_true", help="compare the vectorised packer with shell.host_buffer on every launch (slow)")
    ap.add_argument("--model", choices=sorted(MODELS), default="gpt2-medium")
    a = ap.parse_args()
    CHECK_PACKER[0] = a.check_packer
    global SEQ, HID, HEADS, HEAD, FFN, NORM, ROPE, GATE  # pylint: disable=global-statement
    m = MODELS[a.model]
    SEQ, HID, HEADS, HEAD, FFN = m["seq"], m["hid"], m["heads"], m["head"], m["ffn"]
    NORM, ROPE, GATE = m["norm"], m["rope"], m["gate"]
    os.makedirs(a.out, exist_ok=True)
    dev = None
    if not a.no_device:
        dev = Device(a.xclbin, a.args, a.device, a.runner)
        print(dev.ready, flush=True)
    results = []
    for rep in range(a.reps):
        blk = Block(os.path.join(a.out, f"rep{rep}"), dev, a.seed)
        out = blk.run()
        np.save(os.path.join(a.out, f"block_output_rep{rep}.npy"), out)
        kernel = sum(s["kernel_s"] for s in blk.stats.values())
        load = sum(s["load_s"] for s in blk.stats.values())
        read = sum(s["read_s"] for s in blk.stats.values())
        pack = sum(s["pack_s"] for s in blk.stats.values())
        hostm = sum(s["host_s"] for s in blk.stats.values())
        launches = sum(s["launches"] for s in blk.stats.values())
        row = {"rep": rep, "launches": launches, "wall_s": blk.wall, "kernel_s": kernel, "transfer_s": load + read,
               "pack_s": pack, "host_math_s": hostm, "mismatched_values": sum(blk.mismatch.values()),
               "mismatch_by_stage": blk.mismatch, "stages": blk.stats}
        results.append(row)
        print(f"rep {rep}: {launches} launches, wall {blk.wall:.2f} s, kernel {kernel:.3f} s, transfers {load + read:.3f} s, "
              f"packing {pack:.2f} s, host math {hostm:.3f} s, mismatched values {sum(blk.mismatch.values())}", flush=True)
        for st, s in blk.stats.items():
            if s["launches"]:
                print(f"    {st:6s} {s['launches']:4d} launches  kernel {s['kernel_s']*1e3:9.2f} ms  load {s['load_s']*1e3:8.2f} ms  read {s['read_s']*1e3:8.2f} ms  pack {s['pack_s']*1e3:8.2f} ms", flush=True)
    if dev is not None:
        dev.close()
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8") as h:
        json.dump({"model": a.model, "shape": MODELS[a.model], "seed": a.seed, "device": None if a.no_device else a.xclbin,
                   "semantics": __doc__.split("The frozen semantics")[1], "reps": results}, h, indent=1)
    print("GPT_BLOCK_DONE")


if __name__ == "__main__":
    main()
