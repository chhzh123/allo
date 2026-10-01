# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An xsim bench for Gemmini's whole accelerator, driven by its instructions.

`gen_mxuaccvpu_tb.py` drives Gemmini's mesh, accumulator and scale unit with a
testbench playing the controller, issuing mesh requests back to back. That is
the best any controller could do. This bench runs the same stimulus files on
the `Gemmini` module itself -- reservation station, loop unrollers, load,
store and execute controllers, scratchpad, DMA, TLB -- by issuing the RoCC
commands Gemmini's own library issues:

* `tiled_matmul_auto`'s tiling, reproduced here from `gemmini.h`: the largest
  tiles whose operands fit half the scratchpad and half the accumulator.
* per `tiled_matmul_outer` call, `config_ex`, `config_st` and three
  `config_ld`;
* per tile, one `gemmini_loop_ws` -- six commands. The hardware loop unroller
  turns it into the tile's move-ins, preloads, computes and move-outs.

The CPU is ideal: a command is offered the cycle the last was taken. Memory is
ideal too: a TileLink slave that accepts a request beat every cycle and
answers from the next, a beat a cycle. It is as wide as the system bus of the
Rocket tile Gemmini was elaborated in: 8 bytes in rocket-chip's own system,
which is also VTA's memory port, and 16 in Chipyard's, which widens the bus
to Gemmini's DMA.

Reported, in cycles from the first command to the accelerator going idle:

    total   end to end, loads and stores included
    ex      cycles the execute controller is busy
    ld, st  cycles the load and store controllers are busy
    ex_only cycles the execute controller is busy and neither other is

The three overlap: the unroller loads the next blocks while it computes.

The golden is computed here from the stimulus, in the configuration's own
arithmetic. Gemmini as shipped scales the accumulator by a float32, rounding
to nearest even; the `shift` variant shifts right, as SPMW and VTA do, and
must reproduce the stimulus file's own result.
"""

import argparse
import math
import os
import re
import struct

import numpy as np

BASE = 0x80000000  # the tile's DRAM, so the TLB's address check passes
RELU = 1
WS = 1
K_CONFIG, K_LOOP_WS = 0, 8
K_MVIN2, K_MVIN, K_MVOUT, K_COMPUTE_PRELOADED, K_COMPUTE_ACCUMULATED, K_PRELOAD = (
    1,
    2,
    3,
    4,
    5,
    6,
)
K_MVIN3 = 14
GARBAGE = 0xFFFFFFFF
ACC, ACCUMULATE = 1 << 31, 1 << 30
K_BOUNDS, K_ADDRS_AB, K_ADDRS_DC, K_STRIDES_AB, K_STRIDES_DC = 9, 10, 11, 12, 13
CONFIG_EX, CONFIG_LD, CONFIG_ST = 0, 1, 2


def read_stim(path):
    """The stimulus file: its header and its tensors, by tile."""
    keys, body = {}, {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            key, _, rest = line.partition(" ")
            if key in ("A", "B", "C", "BIAS"):
                body[key] = np.array(rest.split(), dtype=np.int64)
            elif rest.strip():
                keys[key] = int(rest.split()[0])
    if "S" in keys:
        keys["M"] = keys["K"] = keys["N"] = keys["S"]
    M, K, N, T = keys["M"], keys["K"], keys["N"], keys["TILES"]
    A = body["A"].reshape(T, M, K)
    # E3's file has a weight matrix per tile; a layer's has one.
    B = body["B"].reshape(-1, K, N)
    return {
        "M": M,
        "K": K,
        "N": N,
        "tiles": T,
        "shift": keys["SHIFT"],
        "A": A,
        "B": B if len(B) == T else np.repeat(B, T, axis=0),
        "C": body["C"].reshape(T, M, N),
        "bias": body.get("BIAS"),
    }


def params(rtl):
    """`gemmini_params.h` as integers."""
    text = open(os.path.join(rtl, "gemmini_params.h"), encoding="utf-8").read()
    want = ("DIM", "BANK_NUM", "BANK_ROWS", "ACC_ROWS", "MAX_BYTES")
    found = {}
    for name in want:
        found[name] = int(re.search(rf"#define {name} (\d+)", text).group(1))
    return found


def auto_tile(p, I, J, K):
    """`tiled_matmul_auto`'s tile, in blocks, for a weight-stationary matmul."""
    dim = p["DIM"]
    rows = p["BANK_NUM"] * p["BANK_ROWS"]
    max_spad_rows, max_acc_rows = rows // 2, p["ACC_ROWS"] // 2
    db_mats_in_partition = (rows // 2 // 2) // dim
    db_max_tile_i_j = int(math.sqrt((p["ACC_ROWS"] // 2) // dim))
    db_max_tile_k = db_mats_in_partition // db_max_tile_i_j
    ti, tj, tk = min(I, db_max_tile_i_j), min(J, db_max_tile_i_j), min(K, db_max_tile_k)
    spad = lambda i, j, k: (i * k + k * j) * dim
    acc = lambda i, j: i * j * dim
    while True:
        grew = False
        if spad(ti, tj + 1, tk) <= max_spad_rows and acc(ti, tj + 1) <= max_acc_rows:
            if tj + 1 <= J:
                tj, grew = tj + 1, True
        if spad(ti + 1, tj, tk) <= max_spad_rows and acc(ti + 1, tj) <= max_acc_rows:
            if ti + 1 <= I:
                ti, grew = ti + 1, True
        if spad(ti, tj, tk + 1) <= max_spad_rows and tk + 1 <= K:
            tk, grew = tk + 1, True
        if not grew:
            return ti, tj, tk


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def golden(stim, variant):
    """The accelerator's result, in the configuration's own arithmetic."""
    acc = np.einsum("tmk,tkn->tmn", stim["A"], stim["B"])
    # E3's microbenchmark has a bias and a ReLU; a layer's projection neither.
    if stim["bias"] is not None:
        acc = np.maximum(acc + stim["bias"], 0)
    if variant == "shift":
        out = acc >> stim["shift"]
    else:
        scale = np.float32(2.0 ** -stim["shift"])
        out = np.rint(acc.astype(np.float32) * scale).astype(np.int64)
    return np.clip(out, -128, 127)


def image(stim):
    """The memory image, and where each tensor sits in it."""
    M, K, N, T = stim["M"], stim["K"], stim["N"], stim["tiles"]
    align = lambda n: (n + 63) // 64 * 64
    a_base = 64
    b_base = align(a_base + T * M * K)
    d_base = align(b_base + T * K * N)
    c_base = align(d_base + 4 * N)
    mem = np.zeros(align(c_base + T * M * N), dtype=np.uint8)
    mem[a_base : a_base + T * M * K] = (
        stim["A"].astype(np.int8).view(np.uint8).reshape(-1)
    )
    mem[b_base : b_base + T * K * N] = (
        stim["B"].astype(np.int8).view(np.uint8).reshape(-1)
    )
    if stim["bias"] is not None:
        mem[d_base : d_base + 4 * N] = stim["bias"].astype("<i4").view(np.uint8)
    return mem, (a_base, b_base, d_base, c_base)


def configuration(p, stim, variant):
    """`tiled_matmul_outer`'s five configuration commands."""
    dim, K, N = p["DIM"], stim["K"], stim["N"]
    act = RELU if stim["bias"] is not None else 0
    if variant == "shift":
        scale_bits, one_bits = stim["shift"], 0
    else:
        scale_bits, one_bits = f32_bits(2.0 ** -stim["shift"]), f32_bits(1.0)
    ex_rs1 = (one_bits << 32) | (1 << 16) | (act << 3) | (WS << 2) | CONFIG_EX
    cmds = [
        (K_CONFIG, ex_rs1, 1 << 48),
        (K_CONFIG, (act << 2) | CONFIG_ST, (scale_bits << 32) | N),
    ]
    # A's and B's rows; the bias repeats for every row.
    for ident, stride in enumerate((K, N, 0)):
        rs1 = (one_bits << 32) | (dim << 16) | (1 << 8) | (ident << 3) | CONFIG_LD
        cmds.append((K_CONFIG, rs1, stride))
    return cmds


def batched(p, stim, variant, phased=False):
    """Every matmul of the stimulus as one hand-scheduled program.

    `program` is what Gemmini's library issues: a configuration and a loop
    per matmul, each waiting on its own loads. For many small matmuls that is
    not always the best Gemmini can do. This is one configuration, then
    `sp_tiled_matmul_ws`'s command sequence for each matmul in turn -- its
    move-ins, its preloads and computes, its move-out -- with each matmul in
    its own scratchpad and accumulator rows, so the reservation station can
    load the next while it computes this one and stores the last.

    `phased` issues every matmul's move-ins, then every matmul's preloads and
    computes, then every move-out. It is slower end to end, and it is the run
    in which the execute controller never waits for an operand.
    """
    dim = p["DIM"]
    M, K, N, T = stim["M"], stim["K"], stim["N"], stim["tiles"]
    I, J, KB = M // dim, N // dim, K // dim
    has_bias = stim["bias"] is not None
    block_len = p["MAX_BYTES"] // dim
    block_len_acc = max(1, p["MAX_BYTES"] // (dim * 4))
    spad_rows = p["BANK_NUM"] * p["BANK_ROWS"]
    if T * (I * KB + KB * J) * dim > spad_rows or T * I * J * dim > p["ACC_ROWS"]:
        raise SystemExit("the stimulus does not fit the scratchpad in one batch")
    mem, (a_base, b_base, d_base, c_base) = image(stim)
    a_sp = lambda t, i, k: ((t * I + i) * KB + k) * dim
    b_sp = lambda t, k, j: spad_rows - T * KB * J * dim + ((t * KB + k) * J + j) * dim
    c_sp = lambda t, i, j: ((t * I + i) * J + j) * dim
    span = lambda rows, cols, addr: (rows << 48) | (cols << 32) | addr
    loads, work, stores = [], [], []
    for t in range(T):
        A, B = BASE + a_base + t * M * K, BASE + b_base + t * K * N
        C = BASE + c_base + t * M * N
        load, compute, store = [], [], []
        if has_bias:
            for i in range(I):
                for j in range(0, J, block_len_acc):
                    cols = min(block_len_acc, J - j) * dim
                    load.append(
                        (
                            K_MVIN3,
                            BASE + d_base + 4 * j * dim,
                            span(dim, cols, ACC | c_sp(t, i, j)),
                        )
                    )
        for k in range(KB):
            for j in range(0, J, block_len):
                cols = min(block_len, J - j) * dim
                load.append(
                    (K_MVIN2, B + (k * N + j) * dim, span(dim, cols, b_sp(t, k, j)))
                )
        for i in range(I):
            for k in range(0, KB, block_len):
                cols = min(block_len, KB - k) * dim
                load.append(
                    (K_MVIN, A + (i * K + k) * dim, span(dim, cols, a_sp(t, i, k)))
                )
        for k in range(KB):
            for j in range(J):
                for i in range(I):
                    out = ACC | c_sp(t, i, j) | (ACCUMULATE if has_bias or k else 0)
                    pre = b_sp(t, k, j) if i == 0 else GARBAGE
                    compute.append(
                        (K_PRELOAD, span(dim, dim, pre), span(dim, dim, out))
                    )
                    funct = K_COMPUTE_PRELOADED if i == 0 else K_COMPUTE_ACCUMULATED
                    compute.append(
                        (funct, span(dim, dim, a_sp(t, i, k)), span(dim, dim, GARBAGE))
                    )
        for i in range(I):
            for j in range(0, J, block_len):
                cols = min(block_len, J - j) * dim
                store.append(
                    (
                        K_MVOUT,
                        C + (i * N + j) * dim,
                        span(dim, cols, ACC | c_sp(t, i, j)),
                    )
                )
        loads.append(load)
        work.append(compute)
        stores.append(store)
    cmds = configuration(p, stim, variant)
    if phased:
        cmds += [
            cmd for phase in (loads, work, stores) for part in phase for cmd in part
        ]
    else:
        for parts in zip(loads, work, stores):
            cmds += [cmd for part in parts for cmd in part]
    plan = {
        "tile": (I, J, KB),
        "tiles_per_matmul": 1,
        "loops": 0,
        "commands": len(cmds),
        "c_base": c_base,
        "c_bytes": T * M * N,
    }
    return cmds, mem, plan


def program(p, stim, variant):
    """The command list and the memory image, as Gemmini's library issues it."""
    dim = p["DIM"]
    M, K, N, T = stim["M"], stim["K"], stim["N"], stim["tiles"]
    has_bias = stim["bias"] is not None
    act = RELU if has_bias else 0
    mem, (a_base, b_base, d_base, c_base) = image(stim)

    cmds, tiles = [], []
    I, J, KB = M // dim, N // dim, K // dim
    ti, tj, tk = auto_tile(p, I, J, KB)
    I0, J0, K0 = -(-I // ti), -(-J // tj), -(-KB // tk)
    a_reuse, b_reuse = I0 * K0 <= 2, J0 * K0 <= 2
    for t in range(T):
        A, B = BASE + a_base + t * M * K, BASE + b_base + t * K * N
        D, C = BASE + d_base, BASE + c_base + t * M * N
        # tiled_matmul_outer: the configuration, then a loop per tile.
        cmds += configuration(p, stim, variant)
        for i0 in range(I0):
            for j0 in range(J0):
                for k0 in range(K0):
                    a_id = (1 if i0 + k0 == 0 else 2) if a_reuse else 0
                    b_id = (1 if j0 + k0 == 0 else 2) if b_reuse else 0
                    bi = min(ti, I - i0 * ti)
                    bj = min(tj, J - j0 * tj)
                    bk = min(tk, KB - k0 * tk)
                    a = A + i0 * ti * dim * K + k0 * tk * dim
                    b = B + k0 * tk * dim * N + j0 * tj * dim
                    # The bias seeds the accumulator on the first K tile only,
                    # and the result leaves it after the last.
                    d = D + 4 * j0 * tj * dim if has_bias and k0 == 0 else 0
                    c = C + i0 * ti * dim * N + j0 * tj * dim if k0 == K0 - 1 else 0
                    accumulate = 1 if has_bias or k0 else 0
                    cmds += [
                        (K_BOUNDS, 0, (bk << 32) | (bj << 16) | bi),
                        (K_ADDRS_AB, a, b),
                        (K_ADDRS_DC, d, c),
                        (K_STRIDES_AB, K, N),
                        (K_STRIDES_DC, 0, N),
                        (
                            K_LOOP_WS,
                            (a_id << 18) | (b_id << 16) | (act << 8) | accumulate,
                            0,
                        ),
                    ]
                    tiles.append((bi, bj, bk))
    plan = {
        "tile": (ti, tj, tk),
        "tiles_per_matmul": I0 * J0 * K0,
        "loops": len(tiles),
        "commands": len(cmds),
        "c_base": c_base,
        "c_bytes": T * M * N,
    }
    return cmds, mem, plan


def ports(rtl):
    """`Gemmini`'s ports, as ``(direction, width, name)``."""
    text = open(os.path.join(rtl, "Gemmini.sv"), encoding="utf-8").read()
    head = text[
        text.index("module Gemmini(") : text.index(");", text.index("module Gemmini("))
    ]
    found = []
    for line in head.split("\n"):
        hit = re.match(r"\s*(input|output)\s+(?:\[(\d+):0\]\s+)?(\w+)", line)
        if hit:
            found.append((hit.group(1), int(hit.group(2) or 0) + 1, hit.group(3)))
    return found


#: What the bench drives; every other input is tied low.
DRIVEN = {
    "clock": "clock",
    "reset": "reset",
    "io_cmd_valid": "cmd_valid",
    "io_cmd_bits_inst_funct": "cmd_funct",
    "io_cmd_bits_inst_opcode": "7'h7b",
    "io_cmd_bits_inst_xs1": "1'b1",
    "io_cmd_bits_inst_xs2": "1'b1",
    "io_cmd_bits_rs1": "cmd_rs1",
    "io_cmd_bits_rs2": "cmd_rs2",
    "io_cmd_bits_status_prv": "2'd3",
    "io_cmd_bits_status_dprv": "2'd3",
    "io_resp_ready": "1'b1",
    "io_ptw_0_req_ready": "1'b1",
    # Gemmini's DMA translates at user privilege, so with no PMP entry every
    # access is denied. Its bare-metal environment opens all of memory with
    # one NAPOT entry before anything runs; this is that entry.
    "io_ptw_0_pmp_0_cfg_a": "2'd3",
    "io_ptw_0_pmp_0_cfg_r": "1'b1",
    "io_ptw_0_pmp_0_cfg_w": "1'b1",
    "io_ptw_0_pmp_0_cfg_x": "1'b1",
    "io_ptw_0_pmp_0_addr": "{30{1'b1}}",
    "io_ptw_0_pmp_0_mask": "{32{1'b1}}",
    "auto_spad_id_out_a_ready": "1'b1",
    "auto_spad_id_out_d_valid": "d_valid",
    "auto_spad_id_out_d_bits_opcode": "d_opcode",
    "auto_spad_id_out_d_bits_size": "d_size",
    "auto_spad_id_out_d_bits_source": "d_source",
    "auto_spad_id_out_d_bits_data": "d_data",
}
#: The outputs the bench reads.
READ = {
    "io_cmd_ready": "cmd_ready",
    "io_busy": "busy",
    "io_interrupt": "interrupt",
    "auto_spad_id_out_a_valid": "a_valid",
    "auto_spad_id_out_a_bits_opcode": "a_opcode",
    "auto_spad_id_out_a_bits_size": "a_size",
    "auto_spad_id_out_a_bits_source": "a_source",
    "auto_spad_id_out_a_bits_address": "a_address",
    "auto_spad_id_out_a_bits_mask": "a_mask",
    "auto_spad_id_out_a_bits_data": "a_data",
    "auto_spad_id_out_d_ready": "d_ready",
}


def testbench(rtl, plan, mem_bytes, limit):
    conns = []
    for direction, width, name in ports(rtl):
        if direction == "input":
            conns.append(
                f"    .{name}({DRIVEN.get(name, str(width) + chr(39) + 'd0')})"
            )
        else:
            conns.append(f"    .{name}({READ.get(name, '')})")
    widths = {name: width for _, width, name in ports(rtl)}
    source = widths["auto_spad_id_out_a_bits_source"]
    data = widths["auto_spad_id_out_a_bits_data"]
    beat = data // 8  # bytes a beat
    lg = beat.bit_length() - 1
    return f"""// Generated by gemmini_rocc_bench.py -- do not edit.
`timescale 1ns/1ps

module tb;
  localparam int NCMD = {plan['commands']};
  localparam int MEMB = {mem_bytes};
  localparam int CBASE = {plan['c_base']};
  localparam int CBYTES = {plan['c_bytes']};
  localparam int QD = 1024;

  reg clock = 0;
  reg reset = 1;
  always #5 clock = ~clock;

  reg [7:0]  mem [0:MEMB-1];
  reg [7:0]  want [0:CBYTES-1];
  reg [6:0]  c_funct [0:NCMD-1];
  reg [63:0] c_rs1 [0:NCMD-1];
  reg [63:0] c_rs2 [0:NCMD-1];

  // -- the RoCC port: an ideal CPU ------------------------------------------
  reg         cmd_valid = 0;
  reg  [6:0]  cmd_funct = 0;
  reg  [63:0] cmd_rs1 = 0, cmd_rs2 = 0;
  wire        cmd_ready, busy, interrupt;

  // -- the TileLink port: an ideal memory -----------------------------------
  wire        a_valid, d_ready;
  wire [2:0]  a_opcode;
  wire [3:0]  a_size;
  wire [{source - 1}:0]  a_source;
  wire [31:0] a_address;
  wire [{beat - 1}:0]  a_mask;
  wire [{data - 1}:0] a_data;
  reg         d_valid = 0;
  reg  [2:0]  d_opcode = 0;
  reg  [3:0]  d_size = 0;
  reg  [{source - 1}:0]  d_source = 0;
  reg  [{data - 1}:0] d_data = 0;

  // One response per request, answered in order, a beat a cycle.
  reg [2:0]  q_opcode [0:QD-1];
  reg [3:0]  q_size [0:QD-1];
  reg [{source - 1}:0]  q_source [0:QD-1];
  reg [31:0] q_addr [0:QD-1];
  integer q_head = 0, q_tail = 0, d_beat = 0, a_beat = 0;

  function automatic integer beats(input [3:0] size);
    beats = (size <= {lg}) ? 1 : (1 << (size - {lg}));
  endfunction
  function automatic [31:0] aligned(input [31:0] addr, input [3:0] size);
    aligned = (size <= {lg}) ? (addr & ~32'd{beat - 1}) : (addr & ~((32'd1 << size) - 1));
  endfunction
  function automatic [{data - 1}:0] beat_of(input [31:0] addr);
    integer b;
    for (b = 0; b < {beat}; b = b + 1) beat_of[8*b +: 8] = mem[addr - 32'h{BASE:x} + b];
  endfunction

  integer wb;
  reg [31:0] wa;
  always @(posedge clock) if (!reset) begin
    if (a_valid) begin
      if (a_opcode == 3'd4) begin
        q_opcode[q_tail % QD] <= 3'd1;
        q_size[q_tail % QD] <= a_size;
        q_source[q_tail % QD] <= a_source;
        q_addr[q_tail % QD] <= aligned(a_address, a_size);
        q_tail <= q_tail + 1;
      end else begin
        wa = aligned(a_address, a_size) + {beat} * a_beat;
        for (wb = 0; wb < {beat}; wb = wb + 1)
          if (a_mask[wb]) mem[wa - 32'h{BASE:x} + wb] <= a_data[8*wb +: 8];
        if (a_beat == beats(a_size) - 1) begin
          a_beat <= 0;
          q_opcode[q_tail % QD] <= 3'd0;
          q_size[q_tail % QD] <= a_size;
          q_source[q_tail % QD] <= a_source;
          q_addr[q_tail % QD] <= 0;
          q_tail <= q_tail + 1;
        end else a_beat <= a_beat + 1;
      end
    end
  end

  // The D channel: the head response's beats, then the next.
  wire d_fire = d_valid && d_ready;
  integer nb, nh;
  always @(posedge clock) if (!reset) begin
    nb = d_beat; nh = q_head;
    if (d_fire) begin
      if (d_opcode == 3'd0 || d_beat == beats(d_size) - 1) begin nb = 0; nh = q_head + 1; end
      else nb = d_beat + 1;
    end
    d_beat <= nb; q_head <= nh;
    if (nh != q_tail) begin
      d_valid <= 1;
      d_opcode <= q_opcode[nh % QD];
      d_size <= q_size[nh % QD];
      d_source <= q_source[nh % QD];
      d_data <= beat_of(q_addr[nh % QD] + {beat} * nb);
    end else d_valid <= 0;
  end

  Gemmini dut (
{chr(10).join(c + ',' for c in conns)[:-1]}
  );

  // -- the program, and the clock on it ---------------------------------------
  integer cycle = 0, issued = 0, first = -1, last_busy = 0, quiet = 0;
  integer n_ex = 0, n_ld = 0, n_st = 0, n_ex_only = 0, errs = 0, i, first_bad = -1;
  wire ex_busy = dut.ex_controller.io_busy;
  wire ld_busy = dut.load_controller.io_busy;
  wire st_busy = dut.store_controller.io_busy;
  always @(posedge clock) begin
    cycle <= cycle + 1;
    if (!reset && first >= 0) begin
      if (ex_busy) n_ex <= n_ex + 1;
      if (ld_busy) n_ld <= n_ld + 1;
      if (st_busy) n_st <= n_st + 1;
      if (ex_busy && !ld_busy && !st_busy) n_ex_only <= n_ex_only + 1;
      if (busy || cmd_valid) begin last_busy <= cycle; quiet <= 0; end
      else quiet <= quiet + 1;
    end
    if (interrupt) begin $display("GEMROCC INTERRUPT at cycle %0d", cycle); $finish; end
  end

  // A command is taken on the edge where valid and ready were both high
  // before it; the next is put up half a cycle later.
  always @(posedge clock) if (!reset && cmd_valid && cmd_ready) begin
    if (first < 0) first = cycle;
    issued = issued + 1;
  end
  always @(negedge clock) if (!reset) begin
    if (issued < NCMD) begin
      cmd_valid = 1;
      cmd_funct = c_funct[issued];
      cmd_rs1 = c_rs1[issued];
      cmd_rs2 = c_rs2[issued];
    end else cmd_valid = 0;
  end

  initial begin
    $readmemh("mem.hex", mem);
    $readmemh("want.hex", want);
    $readmemh("cmd_funct.hex", c_funct);
    $readmemh("cmd_rs1.hex", c_rs1);
    $readmemh("cmd_rs2.hex", c_rs2);
    repeat (20) @(posedge clock);
    reset = 0;
    while (!(issued == NCMD && quiet > 64) && cycle < {limit}) @(posedge clock);
    for (i = 0; i < CBYTES; i = i + 1)
      if (mem[CBASE + i] !== want[i]) begin
        if (first_bad < 0) first_bad = i;
        errs = errs + 1;
      end
    $display("GEMROCC RESULT errs=%0d total=%0d ex=%0d ld=%0d st=%0d ex_only=%0d commands=%0d loops={plan['loops']} first_bad=%0d timeout=%0d",
             errs, last_busy - first + 1, n_ex, n_ld, n_st, n_ex_only, issued, first_bad,
             (cycle >= {limit}) ? 1 : 0);
    $display("GEMROCC %s", (errs == 0 && issued == NCMD && cycle < {limit}) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--rtl", required=True, help="the split SystemVerilog directory"
    )
    parser.add_argument("--stim", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", default="lean", choices=("lean", "shift"))
    parser.add_argument(
        "--program",
        default="library",
        choices=("library", "batched", "phased"),
        help="the commands Gemmini's library issues, one hand-scheduled batch, "
        "or that batch as loads, then computes, then stores",
    )
    args = parser.parse_args()

    p = params(args.rtl)
    stim = read_stim(args.stim)
    if args.program == "library":
        cmds, mem, plan = program(p, stim, args.variant)
    else:
        cmds, mem, plan = batched(p, stim, args.variant, args.program == "phased")
    want = golden(stim, args.variant)
    if args.variant == "shift" and not np.array_equal(want, stim["C"]):
        raise SystemExit("the shift golden does not reproduce the stimulus file's")
    os.makedirs(args.out, exist_ok=True)
    put = lambda name, rows: open(os.path.join(args.out, name), "w").write(
        "\n".join(rows) + "\n"
    )
    put("mem.hex", [f"{b:02x}" for b in mem])
    put("want.hex", [f"{b & 255:02x}" for b in want.reshape(-1)])
    put("cmd_funct.hex", [f"{f:02x}" for f, _, _ in cmds])
    put("cmd_rs1.hex", [f"{a:016x}" for _, a, _ in cmds])
    put("cmd_rs2.hex", [f"{b:016x}" for _, _, b in cmds])
    rows = stim["tiles"] * stim["M"] * stim["K"] * stim["N"] // p["DIM"] ** 2
    limit = 40 * rows + 400000
    put("tb.sv", [testbench(args.rtl, plan, len(mem), limit)])
    print(
        f"GEMROCC PLAN dim={p['DIM']} M={stim['M']} K={stim['K']} N={stim['N']} "
        f"matmuls={stim['tiles']} program={args.program} tile={plan['tile']} loops={plan['loops']} "
        f"commands={plan['commands']} floor={rows} mem={len(mem)} "
        f"bus={next(w for _, w, n in ports(args.rtl) if n.endswith('a_bits_data'))}"
    )


if __name__ == "__main__":
    main()
