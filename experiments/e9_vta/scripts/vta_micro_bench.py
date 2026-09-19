#!/usr/bin/env python3
"""E3's microbenchmark on VTA -- the workload all three engines can run.

E3's micro is a tiled int8 GEMM with bias, ReLU, a requantising shift and a
clip to int8, all on device.  That is the *intersection* of the three ISAs:
Gemmini and SPMW were measured on it already, and every operation in it is
inside VTA's `minpool / maxpool / add / shift` ALU.  So it is the comparison
that does not favour anyone's design envelope.

What it exposes is the one architectural difference that matters here.
Gemmini folds bias, ReLU, shift and clip into its `AccumulatorScale` output
path, and SPMW folds them into its scale lane, so both pay **nothing** for
the epilogue.  VTA's ALU is a load-store unit over the accumulator
scratchpad: one opcode per instruction, one pass over the data each, and
`Compute.scala` asserts the GEMM and the ALU never fetch a micro-op in the
same cycle, so the passes cannot overlap the matmul either.  This bench runs
the real program -- one GEMM instruction and three ALU instructions -- and
measures what that costs.

Both modules are instantiated over one accumulator model and share the
micro-op port, which is how `Compute` wires them.
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np

DIM = 16
OP_MIN, OP_MAX, OP_ADD, OP_SHR = 0, 1, 2, 3


def read_stim(path):
    t = open(path).read()
    g = lambda k: re.search(rf"^{k} (.*)$", t, re.M).group(1).split()
    s = int(g("S")[0]); tiles = int(g("TILES")[0]); shift = int(g("SHIFT")[0])
    bias = np.array(g("BIAS"), dtype=np.int64)
    A = np.array(g("A"), dtype=np.int64).reshape(tiles, s, s)
    B = np.array(g("B"), dtype=np.int64).reshape(tiles, s, s)
    C = np.array(g("C"), dtype=np.int64).reshape(tiles, s, s)
    return s, tiles, shift, bias, A, B, C


def emit(stim, out):
    s, tiles, shift, bias, A, B, C = read_stim(stim)
    assert s == DIM, f"this bench is built for {DIM}x{DIM}, stimulus says {s}"
    nacc = tiles * DIM

    # The golden, recomputed from the operands so the bench checks E3's file
    # rather than trusting it: bias, then ReLU, then the shift, then the clip.
    acc = np.stack([A[t] @ B[t] for t in range(tiles)]) + bias
    want = np.minimum(np.maximum(acc, 0) >> shift, 127)
    if not np.array_equal(want, C):
        sys.exit(f"recomputed golden differs from the stimulus in "
                 f"{int((want != C).sum())} places -- check the op order")

    os.makedirs(out, exist_ok=True)
    with open(f"{out}/inp.dat", "w") as f:
        for t in range(tiles):
            for r in range(DIM):
                f.write(" ".join(f"{v & 0xFF:02x}" for v in A[t][r]) + "\n")
    with open(f"{out}/wgt.dat", "w") as f:
        for t in range(tiles):
            f.write(" ".join(f"{int(B[t][j][i]) & 0xFF:02x}"
                             for i in range(DIM) for j in range(DIM)) + "\n")
    with open(f"{out}/bias.dat", "w") as f:
        for _ in range(nacc):
            f.write(" ".join(f"{int(v) & 0xFFFFFFFF:08x}" for v in bias) + "\n")
    with open(f"{out}/want.dat", "w") as f:
        for t in range(tiles):
            for r in range(DIM):
                f.write(" ".join(f"{int(v) & 0xFFFFFFFF:08x}"
                                 for v in want[t][r]) + "\n")

    conn = dict(
        g_inp="\n".join(f"    .io_inp_rd_0_data_bits_0_{i}(inp_d[{i}])," for i in range(DIM)),
        g_wgt="\n".join(f"    .io_wgt_rd_0_data_bits_{i}_{j}(wgt_d[{i}][{j}]),"
                        for i in range(DIM) for j in range(DIM)),
        g_accr="\n".join(f"    .io_acc_rd_0_data_bits_0_{i}(acc_d[{i}])," for i in range(DIM)),
        g_accw="\n".join(f"    .io_acc_wr_0_bits_data_0_{i}(g_accwd[{i}])," for i in range(DIM)),
        g_outw="\n".join(f"    .io_out_wr_0_bits_data_0_{i}()," for i in range(DIM)),
        g_inpw="\n".join(f"    .io_inp_wr_0_bits_data_0_{i}()," for i in range(DIM)),
        g_wgtw="\n".join(f"    .io_wgt_wr_0_bits_data_{i}_{j}(),"
                         for i in range(DIM) for j in range(DIM)),
        g_outr="\n".join(f"    .io_out_rd_0_data_bits_0_{i}(8'sd0)," for i in range(DIM)),
        a_accr="\n".join(f"    .io_acc_rd_0_data_bits_0_{i}(acc_d[{i}])," for i in range(DIM)),
        a_accw="\n".join(f"    .io_acc_wr_0_bits_data_0_{i}(a_accwd[{i}])," for i in range(DIM)),
        a_outr="\n".join(f"    .io_out_rd_0_data_bits_0_{i}(8'sd0)," for i in range(DIM)),
        a_outw=",\n".join(f"    .io_out_wr_0_bits_data_0_{i}(a_outwd[{i}])" for i in range(DIM)),
        inp_load="\n".join(f"        inp_d[{i}] <= inpm[inp_ib][{i}];" for i in range(DIM)),
        wgt_load="\n".join(f"        wgt_d[{i}][{j}] <= wgtm[wgt_ib][{i}][{j}];"
                           for i in range(DIM) for j in range(DIM)),
        acc_load="\n".join(f"        acc_d[{i}] <= accm[acc_ib][{i}];" for i in range(DIM)),
        acc_store="\n".join(f"        accm[acc_wi][{i}] <= acc_wd[{i}];" for i in range(DIM)),
        dim=DIM, tiles=tiles, nacc=nacc, shift=shift,
        op_min=OP_MIN, op_max=OP_MAX, op_shr=OP_SHR,
    )
    with open(f"{out}/tb.sv", "w") as f:
        f.write(TB % conn)
    return tiles, nacc


TB = r"""
`timescale 1ns/1ps
module tb;
  localparam DIM = %(dim)d, NT = %(tiles)d, NACC = %(nacc)d, SHIFT = %(shift)d;
  reg clk = 1'b0, rst = 1'b1;
  always #1 clk = ~clk;

  reg  [7:0]  inpm [0:NACC-1][0:DIM-1];
  reg  [7:0]  wgtm [0:NT-1][0:DIM-1][0:DIM-1];
  reg  [31:0] accm [0:NACC-1][0:DIM-1];
  reg  [31:0] bias [0:NACC-1][0:DIM-1];
  reg  [31:0] want [0:NACC-1][0:DIM-1];

  reg  signed [7:0]  inp_d [0:DIM-1];
  reg  signed [7:0]  wgt_d [0:DIM-1][0:DIM-1];
  reg  signed [31:0] acc_d [0:DIM-1];
  // Each unit gets its *own* read-valid, derived from its own request.
  // A shared one fires TensorAlu.scala:287 --
  // `assert(acc.rd.data.valid === (valid_r3 || src_valid_r3))` -- the moment
  // the GEMM's accumulator reads are visible to an idle ALU.
  reg  inp_dv = 0, wgt_dv = 0, uop_dv = 0;
  reg  g_acc_dv = 0, a_acc_dv = 0, g_uop_dv = 0, a_uop_dv = 0;
  reg  [10:0] u0 = 0, u1 = 0; reg [9:0] u2 = 0;

  // `sel` picks which unit owns the shared micro-op and accumulator ports.
  // They never run together -- Compute.scala asserts exactly that -- so a
  // mux is the whole arbitration.
  reg sel = 0;                      // 0 = GEMM, 1 = ALU
  reg g_start = 0, a_start = 0;
  reg [2:0] alu_op = 3'd0; reg [15:0] alu_imm = 16'd0;

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

  integer cyc = 0, t0 = -1, t1 = -1, i, j, k, errs = 0;

  TensorGemm gemm (
    .clock(clk), .reset(rst), .io_start(g_start), .io_done(g_done),
    .io_dec_wgt_1(10'd0), .io_dec_wgt_0(10'd1),
    .io_dec_inp_1(11'd0), .io_dec_inp_0(11'd%(dim)d),
    .io_dec_acc_1(11'd0), .io_dec_acc_0(11'd%(dim)d),
    .io_dec_empty_0(1'b0),
    .io_dec_lp_1(14'd1), .io_dec_lp_0(14'd%(tiles)d),
    .io_dec_uop_end(14'd%(dim)d), .io_dec_uop_begin(13'd0),
    .io_dec_reset(1'b0), .io_dec_push_next(1'b0), .io_dec_push_prev(1'b0),
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
    .io_dec_src_1(11'd0), .io_dec_src_0(11'd%(dim)d),
    .io_dec_dst_1(11'd0), .io_dec_dst_0(11'd%(dim)d),
    .io_dec_empty_0(1'b0),
    .io_dec_lp_1(14'd1), .io_dec_lp_0(14'd%(tiles)d),
    .io_dec_uop_end(14'd%(dim)d), .io_dec_uop_begin(13'd0),
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
    if (uop_iv) begin u0 <= uop_ib[10:0]; u1 <= uop_ib[10:0]; u2 <= 10'd0; end
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

  task automatic run_alu(input [2:0] op, input [15:0] imm);
    begin
      sel = 1; alu_op = op; alu_imm = imm;
      @(posedge clk); a_start = 1; @(posedge clk); a_start = 0;
      for (k = 0; k < 200000; k = k + 1) begin
        @(posedge clk);
        if (a_done) k = 200000;
      end
      @(posedge clk);
    end
  endtask

  initial begin
    $readmemh("inp.dat", inpm);
    $readmemh("wgt.dat", wgtm);
    $readmemh("bias.dat", bias);
    $readmemh("want.dat", want);
    // The bias is the accumulator's initial value, which is how all three
    // engines get it -- Gemmini and SPMW fold it into the epilogue, VTA
    // preloads the accumulator. Nobody is charged a pass for it.
    for (i = 0; i < NACC; i = i + 1)
      for (j = 0; j < DIM; j = j + 1) accm[i][j] = bias[i][j];
    repeat (8) @(posedge clk);
    rst = 0; @(posedge clk);

    sel = 0;
    g_start = 1; @(posedge clk); g_start = 0;
    for (k = 0; k < 200000; k = k + 1) begin
      @(posedge clk);
      if (g_done) k = 200000;
    end
    @(posedge clk);

    run_alu(3'd%(op_max)d, 16'd0);        // ReLU
    run_alu(3'd%(op_shr)d, 16'd%(shift)d); // requantise
    run_alu(3'd%(op_min)d, 16'd127);      // clip to int8

    repeat (8) @(posedge clk);
    for (i = 0; i < NACC; i = i + 1)
      for (j = 0; j < DIM; j = j + 1)
        if (accm[i][j] !== want[i][j]) begin
          if (errs < 8)
            $display("VTAMICRO MISMATCH acc[%%0d][%%0d] got=%%0d want=%%0d",
                     i, j, $signed(accm[i][j]), $signed(want[i][j]));
          errs = errs + 1;
        end
    $display("VTAMICRO RESULT tiles=%%0d errs=%%0d cycles=%%0d per_tile=%%0d",
             NT, errs, (t1 - t0), (t1 - t0) / NT);
    $display("VTAMICRO %%s", (errs == 0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim", required=True)
    ap.add_argument("--gemm-rtl",
                    default="/scratch/hc676/vta/hardware/chisel/vta_out_TensorGemm")
    ap.add_argument("--alu-rtl", default="/scratch/hc676/vta_build/rtl_alu_base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    tiles, nacc = emit(a.stim, a.out)
    print(f"VTAMICRO BENCH tiles={tiles} acc_rows={nacc} -> {a.out}")
    if not a.run:
        return
    v = [os.path.join(d, f) for d in (a.gemm_rtl, a.alu_rtl)
         for f in sorted(os.listdir(d)) if f.endswith(".v")]
    # both trees carry the shared leaf modules, so duplicates are dropped
    seen, files = set(), []
    for f in v:
        b = os.path.basename(f)
        if b not in seen:
            seen.add(b)
            files.append(f)
    for cmd in (["xvlog"] + files, ["xvlog", "-sv", "tb.sv"],
                ["xelab", "tb", "-s", "tbsim", "-timescale", "1ns/1ps",
                 "-L", "unisims_ver", "-L", "unimacro_ver", "-L", "secureip"],
                ["xsim", "tbsim", "-runall"]):
        r = subprocess.run(cmd, cwd=a.out, capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()
        if r.returncode:
            print(f"FAILED: {' '.join(cmd[:2])}")
            print("\n".join(tail[-25:]))
            sys.exit(1)
        for line in tail:
            if line.startswith("VTAMICRO ") or "Error" in line:
                print(line)


if __name__ == "__main__":
    main()
