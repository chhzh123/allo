#!/usr/bin/env python3
"""LLaMA-3.2-1B's gate and up projections on VTA's `TensorGemm` and `TensorAlu`.

The stimulus is `test_spmw_llama_ffn.dump_llama`'s: ``x`` (64 tokens x 2048),
the ``[W_gate | W_up]`` slice (2048 x 128) and the golden
``clip(x W >> s, -128, 127)``. No bias and no activation, so VTA's program is
its GEMM and three ALU passes -- shift, max -128, min 127 -- which is the
whole of what E9's other two engines compute for this layer too.

A layer this deep does not fit VTA's scratchpads the way E3's tiles did. The
elaborated units index 2,048 input rows, 1,024 weight blocks and 2,048
accumulator rows, so the program runs the way a real VTA program is tiled:
per column chunk, a GEMM with ``reset`` set zeroes the accumulator, one GEMM
per chunk of ``K`` blocks accumulates into it, and the three ALU passes
requantise it. A GEMM's loops are ``lp_0`` over the chunk's column blocks
and ``lp_1`` over its ``K`` blocks, and its micro-ops walk the 64 rows, so
every weight block is read once and all 64 tokens stream past it.

What pages through the scratchpads between GEMMs is outside `TensorGemm` and
`TensorAlu` -- VTA's load and store units, which overlap it with compute --
so the bench swaps each page in at no cost, as it hands the other two
engines their operands at no cost. The cycles are the two units' own.
"""

import argparse
import os
import re
import sys

import numpy as np

from vta_micro_bench import OP_MAX, OP_MIN, OP_SHR, ports, rtl_dirs, simulate

INP_ROWS, WGT_BLOCKS, ACC_ROWS = 2048, 1024, 2048  # what the units can index


def read_stim(path):
    """The LLaMA stimulus: shape, shift, ``x``, ``W`` and the golden."""
    t = open(path).read()
    g = lambda k: re.search(rf"^{k} (.*)$", t, re.M).group(1).split()
    m, k, n = (int(g(key)[0]) for key in ("M", "K", "N"))
    shift = int(g("SHIFT")[0])
    A = np.array(g("A"), dtype=np.int64).reshape(m, k)
    B = np.array(g("B"), dtype=np.int64).reshape(k, n)
    C = np.array(g("C"), dtype=np.int64).reshape(m, n)
    return m, k, n, shift, A, B, C


def chunks(L, KB, NB):
    """The largest ``K`` and column chunks the scratchpads hold."""
    kbc = min(KB, INP_ROWS // L)
    nbc = min(NB, WGT_BLOCKS // kbc, ACC_ROWS // L)
    if KB % kbc or NB % nbc:
        sys.exit(f"{KB} x {NB} blocks do not page as {kbc} x {nbc}")
    return kbc, nbc


def emit(stim, out, S):
    L, K, N, shift, A, B, C = read_stim(stim)
    if K % S or N % S:
        sys.exit(f"a {K} x {N} weight does not block onto a {S}-wide VTA")
    want = np.clip((A @ B) >> shift, -128, 127)
    if not np.array_equal(want, C):
        sys.exit(f"recomputed golden differs in {int((want != C).sum())} places")
    KB, NB = K // S, N // S
    kbc, nbc = chunks(L, KB, NB)
    kch = KB // kbc

    os.makedirs(out, exist_ok=True)
    # inp row (kb, m) is x[m, kb*S : kb*S+S], so a K chunk is contiguous.
    with open(f"{out}/inp.dat", "w") as f:
        for kb in range(KB):
            for r in range(L):
                f.write(
                    " ".join(f"{v & 0xFF:02x}" for v in A[r, kb * S : (kb + 1) * S])
                )
                f.write("\n")
    # weight blocks page by page -- (column chunk, K chunk) -- and within a
    # page (kb, nb), as the GEMM's loops index them; entry (i, j) is VTA's
    # [out][in], W[kb*S + j, nb*S + i].
    with open(f"{out}/wgt.dat", "w") as f:
        for cn in range(NB // nbc):
            for ck in range(kch):
                for kl in range(kbc):
                    for nl in range(nbc):
                        kb, nb = ck * kbc + kl, cn * nbc + nl
                        blk = B[kb * S : (kb + 1) * S, nb * S : (nb + 1) * S]
                        f.write(
                            " ".join(
                                f"{int(blk[j, i]) & 0xFF:02x}"
                                for i in range(S)
                                for j in range(S)
                            )
                        )
                        f.write("\n")
    # output row (nb, m) is y[m, nb*S : nb*S+S]: a column chunk is contiguous.
    with open(f"{out}/want.dat", "w") as f:
        for nb in range(NB):
            for r in range(L):
                f.write(
                    " ".join(
                        f"{int(v) & 0xFFFFFFFF:08x}"
                        for v in want[r, nb * S : (nb + 1) * S]
                    )
                )
                f.write("\n")
    # The GEMM's micro-ops: row m reads inp m and writes acc m; the loops add
    # the block offsets.
    with open(f"{out}/uop.dat", "w") as f:
        for r in range(L):
            f.write(f"{(r << 11) | r:08x}\n")
    conn = dict(
        ports(S, inp="inp_base + inp_ib", wgt="wgt_base + wgt_ib"),
        dim=S,
        rows=L,
        shift=shift,
        kb=KB,
        nb=NB,
        kbc=kbc,
        nbc=nbc,
        op_min=OP_MIN,
        op_max=OP_MAX,
        op_shr=OP_SHR,
    )
    with open(f"{out}/tb.sv", "w") as f:
        f.write(TB % conn)
    return L, K, N, kbc, nbc


TB = r"""
`timescale 1ns/1ps
module tb;
  localparam DIM = %(dim)d, L = %(rows)d, SHIFT = %(shift)d;
  localparam KB = %(kb)d, NB = %(nb)d, KBC = %(kbc)d, NBC = %(nbc)d;
  localparam KCH = KB / KBC, NCH = NB / NBC;
  localparam NINP = KB * L, NWGT = KB * NB, NOUT = NB * L, NACC = NBC * L;
  reg clk = 1'b0, rst = 1'b1;
  always #1 clk = ~clk;

  reg  [7:0]  inpm [0:NINP-1][0:DIM-1];
  reg  [7:0]  wgtm [0:NWGT-1][0:DIM-1][0:DIM-1];
  reg  [31:0] uopm [0:L-1];
  reg  [31:0] accm [0:NACC-1][0:DIM-1];
  reg  [31:0] outm [0:NOUT-1][0:DIM-1];
  reg  [31:0] want [0:NOUT-1][0:DIM-1];
  integer inp_base = 0, wgt_base = 0;  // the pages the load unit has swapped in

  reg  signed [7:0]  inp_d [0:DIM-1];
  reg  signed [7:0]  wgt_d [0:DIM-1][0:DIM-1];
  reg  signed [31:0] acc_d [0:DIM-1];
  // Each unit gets its own read-valid, as in vta_micro_bench.
  reg  inp_dv = 0, wgt_dv = 0;
  reg  g_acc_dv = 0, a_acc_dv = 0, g_uop_dv = 0, a_uop_dv = 0;
  reg  [10:0] u0 = 0, u1 = 0; reg [9:0] u2 = 0;

  reg sel = 0;                      // 0 = GEMM, 1 = ALU: never both
  reg g_start = 0, a_start = 0;
  reg [2:0] alu_op = 3'd0; reg [15:0] alu_imm = 16'd0;
  // The GEMM's decoded instruction, held while it runs.
  reg g_reset = 1'b0; reg [13:0] g_lp1 = 14'd1;

  wire g_uop_iv, g_inp_iv, g_wgt_iv, g_acc_iv, g_acc_wv, g_done;
  wire a_uop_iv, a_acc_iv, a_acc_wv, a_done;
  wire [10:0] g_uop_ib, inp_ib, g_acc_ib, g_acc_wi;
  wire [10:0] a_uop_ib, a_acc_ib, a_acc_wi;
  wire [9:0]  wgt_ib;
  wire signed [31:0] g_accwd [0:DIM-1];
  wire signed [31:0] a_accwd [0:DIM-1];
  wire signed [7:0]  a_outwd [0:DIM-1];

  wire        uop_iv  = sel ? a_uop_iv : g_uop_iv;
  wire [10:0] uop_ib  = sel ? a_uop_ib : g_uop_ib;
  wire        acc_iv  = sel ? a_acc_iv : g_acc_iv;
  wire [10:0] acc_ib  = sel ? a_acc_ib : g_acc_ib;
  wire        acc_wv  = sel ? a_acc_wv : g_acc_wv;
  wire [10:0] acc_wi  = sel ? a_acc_wi : g_acc_wi;
  wire signed [31:0] acc_wd [0:DIM-1];
  genvar gi;
  generate for (gi = 0; gi < DIM; gi = gi + 1) begin : wsel
    assign acc_wd[gi] = sel ? a_accwd[gi] : g_accwd[gi];
  end endgenerate

  integer cyc = 0, t0 = -1, t1 = -1, i, j, k, cn, ck, errs = 0, n_instr = 0;
  integer c_gemm = 0, c_reset = 0, c_alu = 0, ts;

  TensorGemm gemm (
    .clock(clk), .reset(rst), .io_start(g_start), .io_done(g_done),
    .io_dec_wgt_1(10'd%(nbc)d), .io_dec_wgt_0(10'd1),
    .io_dec_inp_1(11'd%(rows)d), .io_dec_inp_0(11'd0),
    .io_dec_acc_1(11'd0), .io_dec_acc_0(11'd%(rows)d),
    .io_dec_empty_0(1'b0),
    .io_dec_lp_1(g_lp1), .io_dec_lp_0(14'd%(nbc)d),
    .io_dec_uop_end(14'd%(rows)d), .io_dec_uop_begin(13'd0),
    .io_dec_reset(g_reset), .io_dec_push_next(1'b0), .io_dec_push_prev(1'b0),
    .io_dec_pop_next(1'b0), .io_dec_pop_prev(1'b0), .io_dec_op(3'd0),
    .io_uop_idx_valid(g_uop_iv), .io_uop_idx_bits(g_uop_ib),
    .io_uop_data_valid(g_uop_dv),
    .io_uop_data_bits_u2(u2), .io_uop_data_bits_u1(u1), .io_uop_data_bits_u0(u0),
    .io_inp_rd_0_idx_valid(g_inp_iv), .io_inp_rd_0_idx_bits(inp_ib),
    .io_inp_rd_0_data_valid(inp_dv),
%(g_inp)s
    .io_inp_wr_0_valid(), .io_inp_wr_0_bits_idx(),
%(g_inpw)s
    .io_wgt_rd_0_idx_valid(g_wgt_iv), .io_wgt_rd_0_idx_bits(wgt_ib),
    .io_wgt_rd_0_data_valid(wgt_dv),
%(g_wgt)s
    .io_wgt_wr_0_valid(), .io_wgt_wr_0_bits_idx(),
%(g_wgtw)s
    .io_acc_rd_0_idx_valid(g_acc_iv), .io_acc_rd_0_idx_bits(g_acc_ib),
    .io_acc_rd_0_data_valid(g_acc_dv),
%(g_accr)s
    .io_acc_wr_0_valid(g_acc_wv), .io_acc_wr_0_bits_idx(g_acc_wi),
%(g_accw)s
    .io_out_rd_0_idx_valid(), .io_out_rd_0_idx_bits(),
    .io_out_rd_0_data_valid(1'b0),
%(g_outr)s
    .io_out_wr_0_valid(), .io_out_wr_0_bits_idx(),
%(g_outw)s
    .io_state(), .io_inflight()
  );

  TensorAlu alu (
    .clock(clk), .reset(rst), .io_start(a_start), .io_done(a_done),
    .io_dec_alu_imm(alu_imm), .io_dec_alu_use_imm(1'b1), .io_dec_alu_op(alu_op),
    .io_dec_src_1(11'd0), .io_dec_src_0(11'd%(rows)d),
    .io_dec_dst_1(11'd0), .io_dec_dst_0(11'd%(rows)d),
    .io_dec_empty_0(1'b0),
    .io_dec_lp_1(14'd1), .io_dec_lp_0(14'd%(nbc)d),
    .io_dec_uop_end(14'd%(rows)d), .io_dec_uop_begin(13'd0),
    .io_dec_reset(1'b0), .io_dec_push_next(1'b0), .io_dec_push_prev(1'b0),
    .io_dec_pop_next(1'b0), .io_dec_pop_prev(1'b0), .io_dec_op(3'd0),
    .io_uop_idx_valid(a_uop_iv), .io_uop_idx_bits(a_uop_ib),
    .io_uop_data_valid(a_uop_dv),
    .io_uop_data_bits_u2(u2), .io_uop_data_bits_u1(u1), .io_uop_data_bits_u0(u0),
    .io_acc_rd_0_idx_valid(a_acc_iv), .io_acc_rd_0_idx_bits(a_acc_ib),
    .io_acc_rd_0_data_valid(a_acc_dv),
%(a_accr)s
    .io_acc_wr_0_valid(a_acc_wv), .io_acc_wr_0_bits_idx(a_acc_wi),
%(a_accw)s
    .io_out_rd_0_idx_valid(), .io_out_rd_0_idx_bits(),
    .io_out_rd_0_data_valid(1'b0),
%(a_outr)s
    .io_out_wr_0_valid(), .io_out_wr_0_bits_idx(),
%(a_outw)s
  );

  always @(posedge clk) begin
    g_uop_dv <= g_uop_iv;  a_uop_dv <= a_uop_iv;
    if (uop_iv) begin
      if (sel) begin
        u0 <= uop_ib[10:0]; u1 <= uop_ib[10:0]; u2 <= 10'd0;
      end else begin
        u0 <= uopm[uop_ib][10:0]; u1 <= uopm[uop_ib][21:11]; u2 <= uopm[uop_ib][31:22];
      end
    end
    inp_dv <= g_inp_iv;
    if (g_inp_iv) begin
%(inp_load)s
    end
    wgt_dv <= g_wgt_iv;
    if (g_wgt_iv) begin
%(wgt_load)s
    end
    g_acc_dv <= g_acc_iv;  a_acc_dv <= a_acc_iv;
    if (acc_iv) begin
%(acc_load)s
    end
    if (acc_wv) begin
%(acc_store)s
      t1 = cyc;
    end
    if (!rst) cyc = cyc + 1;
    if (!rst && uop_iv && t0 < 0) t0 = cyc;
  end

  task automatic run_gemm(input rs, input [13:0] lp1);
    begin
      sel = 0; g_reset = rs; g_lp1 = lp1; ts = cyc; n_instr = n_instr + 1;
      @(posedge clk); g_start = 1; @(posedge clk); g_start = 0;
      for (k = 0; k < 100000000; k = k + 1) begin
        @(posedge clk);
        if (g_done) k = 100000000;
      end
      @(posedge clk);
      if (rs) c_reset = c_reset + (cyc - ts); else c_gemm = c_gemm + (cyc - ts);
    end
  endtask

  task automatic run_alu(input [2:0] op, input [15:0] imm);
    begin
      sel = 1; alu_op = op; alu_imm = imm; ts = cyc; n_instr = n_instr + 1;
      @(posedge clk); a_start = 1; @(posedge clk); a_start = 0;
      for (k = 0; k < 100000000; k = k + 1) begin
        @(posedge clk);
        if (a_done) k = 100000000;
      end
      @(posedge clk);
      c_alu = c_alu + (cyc - ts);
    end
  endtask

  initial begin
    $readmemh("inp.dat", inpm);
    $readmemh("wgt.dat", wgtm);
    $readmemh("want.dat", want);
    $readmemh("uop.dat", uopm);
    repeat (8) @(posedge clk);
    rst = 0; @(posedge clk);

    for (cn = 0; cn < NCH; cn = cn + 1) begin
      run_gemm(1'b1, 14'd1);                   // zero this chunk's accumulator
      for (ck = 0; ck < KCH; ck = ck + 1) begin
        inp_base = ck * KBC * L;
        wgt_base = (cn * KCH + ck) * KBC * NBC;
        run_gemm(1'b0, 14'd%(kbc)d);
      end
      run_alu(3'd%(op_shr)d, SHIFT);           // requantise
      run_alu(3'd%(op_max)d, 16'hff80);        // clip at -128
      run_alu(3'd%(op_min)d, 16'd127);         // and at 127
      // The store unit's work, outside the measured units.
      for (i = 0; i < NACC; i = i + 1)
        for (j = 0; j < DIM; j = j + 1) outm[cn * NACC + i][j] = accm[i][j];
    end

    repeat (8) @(posedge clk);
    for (i = 0; i < NOUT; i = i + 1)
      for (j = 0; j < DIM; j = j + 1)
        if (outm[i][j] !== want[i][j]) begin
          if (errs < 8)
            $display("VTALLAMA MISMATCH out[%%0d][%%0d] got=%%0d want=%%0d",
                     i, j, $signed(outm[i][j]), $signed(want[i][j]));
          errs = errs + 1;
        end
    $display("VTALLAMA RESULT errs=%%0d cycles=%%0d gemm=%%0d reset=%%0d alu=%%0d instructions=%%0d pages=%%0dx%%0d",
             errs, (t1 - t0), c_gemm, c_reset, c_alu, n_instr, KCH, NCH);
    $display("VTALLAMA %%s", (errs == 0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim", required=True)
    ap.add_argument("--width", type=int, required=True, help="the VTA's width")
    ap.add_argument("--gemm-rtl", default=None)
    ap.add_argument("--alu-rtl", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    L, K, N, kbc, nbc = emit(a.stim, a.out, a.width)
    print(
        f"VTALLAMA BENCH m={L} k={K} n={N} width={a.width} pages={kbc}x{nbc} -> {a.out}"
    )
    if a.run:
        simulate(a.out, *rtl_dirs(a.width, a.gemm_rtl, a.alu_rtl), tag="VTALLAMA ")


if __name__ == "__main__":
    main()
