#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An xsim bench for VTA's whole engine, `Core`, running a real program.

`vta_micro_bench.py` and `vta_llama_bench.py` drive `TensorGemm` and
`TensorAlu` with decoded instructions and hold VTA's scratchpads in the
testbench, paging them at no cost. That counts the two units' own cycles and
nothing of fetch, decode, load or store. This bench launches `Core` on a
128-bit instruction stream in memory, as VTA's runtime does, and counts from
launch to finish:

* fetch reads the instructions and routes them to the three units' queues;
* LOAD instructions bring micro-ops, inputs, weights and biases in from
  memory, and a STORE takes the result out;
* GEMM and ALU instructions run through `Compute`'s decode;
* the units keep step with VTA's dependency tokens.

Memory is ideal: a VME client that takes a command whenever it has room and
returns a 64-bit beat every cycle from the next.

The programs are the best VTA allows, not a literal port:

* **A layer** is paged through the scratchpads in two halves. Page ``p``
  loads into half ``p % 2`` while the GEMM of page ``p - 1`` reads the other,
  so loads hide behind compute -- VTA's two-virtual-thread schedule. A GEMM
  frees its half for the page after next with a token.
* **E3's microbenchmark** gets its bias from one accumulator row per column
  block and an ALU `add`, not from loading a whole accumulator image, which
  costs four to eight times as many memory beats as the pass does cycles.

Reported, in cycles from launch to finish:

    total   end to end
    gemm    cycles `Compute` spends executing GEMM instructions
    alu     ... ALU instructions
    cload   ... its own loads: micro-ops and accumulator rows
    load    cycles the load unit is busy
    store   cycles the store unit is busy
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np

OP_LOAD, OP_STORE, OP_GEMM, OP_FINISH, OP_ALU = 0, 1, 2, 3, 4
ID_UOP, ID_WGT, ID_INP, ID_ACC, ID_OUT = 0, 1, 2, 3, 4
ALU_MIN, ALU_MAX, ALU_ADD, ALU_SHR = 0, 1, 2, 3
INP_ROWS, WGT_BLOCKS, ACC_ROWS = 2048, 1024, 2048  # the shipped scratchpads
BASE = {
    "ins": 0x00000000,
    "uop": 0x01000000,
    "inp": 0x02000000,
    "wgt": 0x04000000,
    "acc": 0x08000000,
    "out": 0x0A000000,
}
#: `Core`'s read channels, in its own order.
READS = ("ins", "uop", "inp", "wgt", "acc")


def deps(pop_prev=0, pop_next=0, push_prev=0, push_next=0):
    return (pop_prev << 3) | (pop_next << 4) | (push_prev << 5) | (push_next << 6)


def mem(op, ident, sram, dram, xsize, dep=0):
    """A LOAD or STORE of ``xsize`` tensors in one row."""
    return (
        op
        | dep
        | (ident << 7)
        | (sram << 10)
        | (dram << 26)
        | (1 << 64)  # ysize
        | (xsize << 80)
        | (xsize << 96)  # xstride
    )


def gemm(
    ub, ue, lp0, lp1, acc0=0, acc1=0, inp0=0, inp1=0, wgt0=0, wgt1=0, reset=0, dep=0
):
    return (
        OP_GEMM
        | dep
        | (reset << 7)
        | (ub << 8)
        | (ue << 21)
        | (lp0 << 35)
        | (lp1 << 49)
        | (acc0 << 64)
        | (acc1 << 75)
        | (inp0 << 86)
        | (inp1 << 97)
        | (wgt0 << 108)
        | (wgt1 << 118)
    )


def alu(op, ub, ue, lp0, dst0, src0=None, imm=None, dep=0):
    """An ALU pass; with ``imm`` None the operand is the source tensor."""
    src0 = dst0 if src0 is None else src0
    use_imm = 0 if imm is None else 1
    return (
        OP_ALU
        | dep
        | (ub << 8)
        | (ue << 21)
        | (lp0 << 35)
        | (1 << 49)  # lp1
        | (dst0 << 64)
        | (src0 << 86)
        | (op << 108)
        | (use_imm << 111)
        | (((imm or 0) & 0xFFFF) << 112)
    )


def uop(u0, u1=0, u2=0):
    return (u2 << 22) | (u1 << 11) | u0


def read_stim(path):
    """The stimulus file: shape, shift, tensors by tile and the bias."""
    text = open(path, encoding="utf-8").read()
    get = lambda key: re.search(rf"^{key} (.*)$", text, re.M)
    val = lambda key: int(get(key).group(1).split()[0])
    arr = lambda key: np.array(get(key).group(1).split(), dtype=np.int64)
    if get("S"):
        m = k = n = val("S")
    else:
        m, k, n = val("M"), val("K"), val("N")
    tiles = val("TILES")
    return {
        "M": m,
        "K": k,
        "N": n,
        "tiles": tiles,
        "shift": val("SHIFT"),
        "A": arr("A").reshape(tiles, m, k),
        "B": arr("B").reshape(tiles, k, n),
        "C": arr("C").reshape(tiles, m, n),
        "bias": arr("BIAS") if get("BIAS") else None,
    }


def blocks(B, S):
    """A weight matrix as VTA's blocks: ``[kb][nb]`` of ``[out][in]`` bytes."""
    K, N = B.shape
    out = np.zeros((K // S, N // S, S, S), dtype=np.int64)
    for kb in range(K // S):
        for nb in range(N // S):
            out[kb, nb] = B[kb * S : (kb + 1) * S, nb * S : (nb + 1) * S].T
    return out


def micro(stim, S):
    """E3's microbenchmark: every tile in one GEMM, the epilogue in ALU passes."""
    M, tiles, bias = stim["M"], stim["tiles"], stim["bias"]
    KB, NB = stim["K"] // S, stim["N"] // S
    nbm, kbm, kbnb = NB * M, KB * M, KB * NB
    nacc, ninp, nwgt, nuop = tiles * nbm, tiles * kbm, tiles * kbnb, NB * KB * M
    if max(ninp, nacc + NB) > INP_ROWS or nwgt > WGT_BLOCKS:
        sys.exit("the microbenchmark does not fit the scratchpads at this width")
    # inp row (t, kb, m), weight block (t, kb, nb), accumulator row (t, nb, m).
    inp = np.stack(
        [
            stim["A"][t][:, kb * S : (kb + 1) * S]
            for t in range(tiles)
            for kb in range(KB)
        ]
    ).reshape(ninp, S)
    wgt = np.stack([blocks(stim["B"][t], S) for t in range(tiles)]).reshape(nwgt, S * S)
    acc = bias.reshape(NB, S)  # one row per column block, kept past the tiles' rows
    uops = [
        uop(nb * M + r, kb * M + r, kb * NB + nb)
        for nb in range(NB)
        for kb in range(KB)
        for r in range(M)
    ]
    # Longer ALU iterations do not help: the pass costs 1.26 cycles a row
    # however the rows are grouped, and a longer micro-op table costs more to
    # load. So the passes loop over the tiles as the GEMM does.
    ident = len(uops)  # row r of a tile onto itself
    uops += [uop(r, r) for r in range(nbm)]
    badd = len(uops)  # row (nb, m) and its column block's bias row
    uops += [uop(nb * M + r, nacc + nb) for nb in range(NB) for r in range(M)]
    over = dict(lp0=tiles, dst0=nbm)
    ins = [
        mem(OP_LOAD, ID_UOP, 0, 0, len(uops)),
        mem(OP_LOAD, ID_ACC, nacc, 0, NB),
        mem(OP_LOAD, ID_INP, 0, 0, ninp),
        mem(OP_LOAD, ID_WGT, 0, 0, nwgt, deps(push_next=1)),
        gemm(ident, ident + nbm, tiles, 1, acc0=nbm, reset=1),
        gemm(0, nuop, tiles, 1, acc0=nbm, inp0=kbm, wgt0=kbnb, dep=deps(pop_prev=1)),
        alu(ALU_ADD, badd, badd + nbm, src0=0, **over),
        alu(ALU_MAX, ident, ident + nbm, imm=0, **over),
        alu(ALU_SHR, ident, ident + nbm, imm=stim["shift"], **over),
        alu(ALU_MIN, ident, ident + nbm, imm=127, dep=deps(push_next=1), **over),
        mem(OP_STORE, ID_OUT, 0, 0, nacc, deps(pop_prev=1, push_prev=1)),
        OP_FINISH | deps(pop_next=1),
    ]
    acc_full = np.einsum("tmk,tkn->tmn", stim["A"], stim["B"]) + bias
    want = np.minimum(np.maximum(acc_full, 0) >> stim["shift"], 127)
    # out row (t, nb, m)
    out = np.stack(
        [want[t][:, nb * S : (nb + 1) * S] for t in range(tiles) for nb in range(NB)]
    ).reshape(nacc, S)
    return ins, uops, inp, wgt, acc, out, {"pages": 1}


def layer(stim, S):
    """A layer's projection, paged through the scratchpads' two halves."""
    L, K, N = stim["M"], stim["K"], stim["N"]
    KB, NB = K // S, N // S
    half_inp, half_wgt = INP_ROWS // 2, WGT_BLOCKS // 2
    kbc = min(KB, half_inp // L)
    nbc = min(NB, half_wgt // kbc, ACC_ROWS // L)
    if KB % kbc or nbc != NB:
        sys.exit(f"{KB} x {NB} blocks do not page as {kbc} x {nbc}")
    pages = KB // kbc
    A, B = stim["A"][0], blocks(stim["B"][0], S)
    # inp row (kb, m) and weight block (kb, nb): a page is contiguous in both.
    inp = np.stack([A[:, kb * S : (kb + 1) * S] for kb in range(KB)]).reshape(KB * L, S)
    wgt = B.reshape(KB * NB, S * S)
    # Row m for each half: the half's own input rows and weight blocks.
    uops = [uop(r, h * half_inp + r, h * half_wgt) for h in range(2) for r in range(L)]
    over = dict(lp0=nbc, dst0=L)
    ins = [
        mem(OP_LOAD, ID_UOP, 0, 0, len(uops)),
        gemm(0, L, nbc, 1, acc0=L, reset=1),
    ]
    for p in range(pages):
        h = p % 2
        # The page before last read this half; its GEMM's token frees it.
        wait = deps(pop_next=1) if p >= 2 else 0
        ins += [
            mem(OP_LOAD, ID_INP, h * half_inp, p * kbc * L, kbc * L, wait),
            mem(
                OP_LOAD,
                ID_WGT,
                h * half_wgt,
                p * kbc * nbc,
                kbc * nbc,
                deps(push_next=1),
            ),
            gemm(
                h * L,
                (h + 1) * L,
                nbc,
                kbc,
                acc0=L,
                inp1=L,
                wgt0=1,
                wgt1=nbc,
                dep=deps(pop_prev=1, push_prev=1 if p + 2 < pages else 0),
            ),
        ]
    ins += [
        alu(ALU_SHR, 0, L, imm=stim["shift"], **over),
        alu(ALU_MAX, 0, L, imm=-128, **over),
        alu(ALU_MIN, 0, L, imm=127, dep=deps(push_next=1), **over),
        mem(OP_STORE, ID_OUT, 0, 0, nbc * L, deps(pop_prev=1, push_prev=1)),
        OP_FINISH | deps(pop_next=1),
    ]
    want = np.clip((A @ stim["B"][0]) >> stim["shift"], -128, 127)
    # out row (nb, m)
    out = np.stack([want[:, nb * S : (nb + 1) * S] for nb in range(NB)]).reshape(
        NB * L, S
    )
    acc = np.zeros((1, S), dtype=np.int64)  # no bias: nothing is loaded
    return ins, uops, inp, wgt, acc, out, {"pages": pages}


def words(data, width):
    """Little-endian values of ``width`` bytes each, as 64-bit hex words."""
    raw = b"".join(int(v).to_bytes(width, "little", signed=False) for v in data)
    raw += b"\0" * (-len(raw) % 8)
    return [raw[i : i + 8][::-1].hex() for i in range(0, len(raw), 8)]


def emit(stim_path, out, S):
    stim = read_stim(stim_path)
    if stim["K"] % S or stim["N"] % S:
        sys.exit(
            f"a {stim['K']} x {stim['N']} weight does not block onto a {S}-wide VTA"
        )
    build = micro if stim["bias"] is not None else layer
    ins, uops, inp, wgt, acc, want, info = build(stim, S)
    os.makedirs(out, exist_ok=True)
    image = {
        "ins": words(ins, 16),
        "uop": words(uops, 4),
        "inp": words(inp.reshape(-1) & 0xFF, 1),
        "wgt": words(wgt.reshape(-1) & 0xFF, 1),
        "acc": words(acc.reshape(-1) & 0xFFFFFFFF, 4),
    }
    for name, rows in image.items():
        with open(f"{out}/{name}.hex", "w", encoding="utf-8") as handle:
            handle.write("\n".join(rows) + "\n")
    with open(f"{out}/want.hex", "w", encoding="utf-8") as handle:
        handle.write("\n".join(f"{int(v) & 255:02x}" for v in want.reshape(-1)) + "\n")
    floor = stim["tiles"] * stim["M"] * stim["K"] * stim["N"] // (S * S)
    with open(f"{out}/tb.sv", "w", encoding="utf-8") as handle:
        handle.write(testbench(image, len(ins), want.size, 60 * floor + 400000))
    print(
        f"VTACORE PLAN width={S} M={stim['M']} K={stim['K']} N={stim['N']} "
        f"tiles={stim['tiles']} instructions={len(ins)} uops={len(uops)} "
        f"pages={info['pages']} floor={floor}"
    )


def testbench(image, n_ins, n_out, limit):
    regions = "\n".join(
        f"  reg [63:0] m_{name} [0:{max(len(rows), 1) - 1}];"
        for name, rows in image.items()
    )
    loads = "\n".join(f'    $readmemh("{name}.hex", m_{name});' for name in image)
    channels, conns = [], []
    for c, name in enumerate(READS):
        channels.append(
            f"""
  // read channel {c}: {name}
  wire        r{c}_cmd_valid, r{c}_data_ready;
  wire [63:0] r{c}_addr;
  wire [7:0]  r{c}_len;
  wire [20:0] r{c}_tag;
  reg  [63:0] r{c}_qaddr [0:QD-1];
  reg  [7:0]  r{c}_qlen [0:QD-1];
  reg  [20:0] r{c}_qtag [0:QD-1];
  integer r{c}_head = 0, r{c}_tail = 0, r{c}_beat = 0, r{c}_beats = 0;
  wire        r{c}_cmd_ready = (r{c}_tail - r{c}_head) < QD;
  wire        r{c}_data_valid = r{c}_head != r{c}_tail;
  wire [63:0] r{c}_word = (r{c}_qaddr[r{c}_head % QD] - 64'h{BASE[name]:x}) >> 3;
  wire [63:0] r{c}_data = m_{name}[r{c}_word + r{c}_beat];
  wire        r{c}_last = r{c}_beat == r{c}_qlen[r{c}_head % QD];
  always @(posedge clock) if (!reset) begin
    if (r{c}_cmd_valid && r{c}_cmd_ready) begin
      r{c}_qaddr[r{c}_tail % QD] <= r{c}_addr;
      r{c}_qlen[r{c}_tail % QD] <= r{c}_len;
      r{c}_qtag[r{c}_tail % QD] <= r{c}_tag;
      r{c}_tail <= r{c}_tail + 1;
    end
    if (r{c}_data_valid && r{c}_data_ready) begin
      r{c}_beats <= r{c}_beats + 1;
      if (r{c}_last) begin r{c}_beat <= 0; r{c}_head <= r{c}_head + 1; end
      else r{c}_beat <= r{c}_beat + 1;
    end
  end"""
        )
        conns.append(
            f"""    .io_vme_rd_{c}_cmd_ready(r{c}_cmd_ready), .io_vme_rd_{c}_cmd_valid(r{c}_cmd_valid),
    .io_vme_rd_{c}_cmd_bits_addr(r{c}_addr), .io_vme_rd_{c}_cmd_bits_len(r{c}_len),
    .io_vme_rd_{c}_cmd_bits_tag(r{c}_tag),
    .io_vme_rd_{c}_data_ready(r{c}_data_ready), .io_vme_rd_{c}_data_valid(r{c}_data_valid),
    .io_vme_rd_{c}_data_bits_data(r{c}_data), .io_vme_rd_{c}_data_bits_tag(r{c}_qtag[r{c}_head % QD]),
    .io_vme_rd_{c}_data_bits_last(r{c}_last),"""
        )
    return f"""// Generated by vta_core_bench.py -- do not edit.
`timescale 1ns/1ps

module tb;
  localparam int QD = 64;
  localparam int NOUT = {n_out};
  reg clock = 0, reset = 1, launch = 0;
  always #5 clock = ~clock;

{regions}
  reg [63:0] m_out [0:{(n_out + 7) // 8}];
  reg [7:0]  want [0:NOUT-1];
{"".join(channels)}

  // the write channel: out
  wire        w_cmd_valid, w_data_valid;
  wire [63:0] w_addr, w_data;
  wire [7:0]  w_len, w_strb;
  reg         w_busy = 0, w_ack = 0;
  reg  [63:0] w_base = 0;
  reg  [7:0]  w_left = 0;
  integer w_beat = 0, wb;
  always @(posedge clock) if (!reset) begin
    w_ack <= 0;
    if (w_cmd_valid && !w_busy) begin
      w_busy <= 1; w_base <= (w_addr - 64'h{BASE['out']:x}) >> 3; w_left <= w_len; w_beat <= 0;
    end
    if (w_busy && w_data_valid) begin
      for (wb = 0; wb < 8; wb = wb + 1)
        if (w_strb[wb]) m_out[w_base + w_beat][8*wb +: 8] <= w_data[8*wb +: 8];
      if (w_beat == w_left) begin w_busy <= 0; w_ack <= 1; end
      else w_beat <= w_beat + 1;
    end
  end

  wire finish;
  Core dut (
    .clock(clock), .reset(reset),
    .io_vcr_launch(launch), .io_vcr_finish(finish),
    .io_vcr_ecnt_0_valid(), .io_vcr_ecnt_0_bits(),
    .io_vcr_vals_0(32'd{n_ins}),
    .io_vcr_ptrs_0(64'h{BASE['ins']:x}), .io_vcr_ptrs_1(64'h{BASE['uop']:x}),
    .io_vcr_ptrs_2(64'h{BASE['inp']:x}), .io_vcr_ptrs_3(64'h{BASE['wgt']:x}),
    .io_vcr_ptrs_4(64'h{BASE['acc']:x}), .io_vcr_ptrs_5(64'h{BASE['out']:x}),
    .io_vcr_ucnt_0_valid(), .io_vcr_ucnt_0_bits(),
{chr(10).join(conns)}
    .io_vme_wr_0_cmd_ready(!w_busy), .io_vme_wr_0_cmd_valid(w_cmd_valid),
    .io_vme_wr_0_cmd_bits_addr(w_addr), .io_vme_wr_0_cmd_bits_len(w_len),
    .io_vme_wr_0_cmd_bits_tag(),
    .io_vme_wr_0_data_ready(w_busy), .io_vme_wr_0_data_valid(w_data_valid),
    .io_vme_wr_0_data_bits_data(w_data), .io_vme_wr_0_data_bits_strb(w_strb),
    .io_vme_wr_0_ack(w_ack)
  );

  integer cycle = 0, t0 = -1, t1 = -1, errs = 0, i, first_bad = -1;
  integer n_gemm = 0, n_alu = 0, n_cload = 0, n_load = 0, n_store = 0;
  wire exe = dut.compute.state == 2'd2;
  always @(posedge clock) begin
    cycle <= cycle + 1;
    if (t0 >= 0 && t1 < 0) begin
      if (exe && dut.compute.dec_io_isGemm) n_gemm <= n_gemm + 1;
      if (exe && dut.compute.dec_io_isAlu) n_alu <= n_alu + 1;
      if (exe && (dut.compute.dec_io_isLoadUop || dut.compute.dec_io_isLoadAcc)) n_cload <= n_cload + 1;
      if (dut.load.state != 2'd0) n_load <= n_load + 1;
      if (dut.store.state != 2'd0) n_store <= n_store + 1;
      if (finish) t1 = cycle;
    end
  end

  initial begin
{loads}
    $readmemh("want.hex", want);
    repeat (20) @(posedge clock);
    reset = 0;
    repeat (4) @(posedge clock);
    @(negedge clock); launch = 1; t0 = cycle;
    while (t1 < 0 && cycle < {limit}) @(posedge clock);
    repeat (4) @(posedge clock);
    for (i = 0; i < NOUT; i = i + 1)
      if (m_out[i / 8][8 * (i % 8) +: 8] !== want[i]) begin
        if (first_bad < 0) first_bad = i;
        errs = errs + 1;
      end
    $display("VTACORE RESULT errs=%0d total=%0d gemm=%0d alu=%0d cload=%0d load=%0d store=%0d instructions={n_ins} beats=%0d first_bad=%0d timeout=%0d",
             errs, t1 - t0, n_gemm, n_alu, n_cload, n_load, n_store,
             r0_beats + r1_beats + r2_beats + r3_beats + r4_beats, first_bad, (t1 < 0) ? 1 : 0);
    $display("VTACORE %s", (errs == 0 && t1 >= 0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
"""


def simulate(out, rtl, tag):
    """Compile `Core` and the bench and run it; registers start at zero."""
    script = f"""set -e
cd {out}
xvlog -d RANDOMIZE_REG_INIT -d RANDOMIZE_MEM_INIT -d "RANDOM=32'h0" {rtl}/Core.v > xvlog.log 2>&1
xvlog -sv tb.sv >> xvlog.log 2>&1
xelab tb -s tbsim -timescale 1ns/1ps > xelab.log 2>&1
xsim tbsim -runall > xsim.log 2>&1
rm -rf xsim.dir tbsim.wdb
"""
    done = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )
    if done.returncode:
        print(f"VTACORE {tag} tool failure:\n{done.stdout}{done.stderr}")
        return 1
    log = open(f"{out}/xsim.log", encoding="utf-8").read()
    for line in log.splitlines():
        if line.startswith("VTACORE"):
            print(f"{tag} {line}")
    return 0 if "VTACORE PASS" in log else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--stim", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--rtl", required=True, help="the directory holding Core.v")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    emit(args.stim, args.out, args.width)
    sys.exit(simulate(args.out, args.rtl, args.tag or f"w{args.width}"))


if __name__ == "__main__":
    main()
