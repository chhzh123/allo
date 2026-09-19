#!/usr/bin/env python3
"""An xsim bench for VTA's tensor ALU -- the passes it *can* run on device.

`TensorAlu`'s whole operation set is `min, max, add, shr`, so of this block's
eleven scale-path passes it can do the four plain requantisations (an
arithmetic shift) and could do a ReLU (max against zero).  It cannot do
LayerNorm, softmax or IGELU: those need a divide, a square root, a reciprocal
and two second-order polynomials, and there is no multiplier in the unit at
all.  This measures the rate of the ones it can.

The instruction shape is the GEMM's: `lp_0` and `lp_1` around a micro-op
range, one accumulator row of 16 per micro-op.  `alu_use_imm` with
`alu_op = SHR` is the requantisation.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

DIM = 16
ALU_SHR = 3          # min:0, max:1, add:2, shr:3 -- vta.core ALU_OP order


def emit(nrow, shift, out):
    rng = np.random.default_rng(23)
    acc = rng.integers(-(1 << 20), 1 << 20, size=(nrow, DIM)).astype(np.int64)
    want = acc >> shift                      # arithmetic shift, as the ALU does
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/acc.dat", "w") as f:
        for r in acc:
            f.write(" ".join(f"{int(v) & 0xFFFFFFFF:08x}" for v in r) + "\n")
    with open(f"{out}/want.dat", "w") as f:
        for r in want:
            f.write(" ".join(f"{int(v) & 0xFFFFFFFF:08x}" for v in r) + "\n")
    tb = TB % dict(
        dim=DIM, nrow=nrow, shift=shift, op=ALU_SHR,
        acc_rd="\n".join(f"    .io_acc_rd_0_data_bits_0_{i}(acc_d[{i}])," for i in range(DIM)),
        acc_wr="\n".join(f"    .io_acc_wr_0_bits_data_0_{i}(acc_wd[{i}])," for i in range(DIM)),
        out_wr=",\n".join(f"    .io_out_wr_0_bits_data_0_{i}(out_wd[{i}])" for i in range(DIM)),
        out_rd_tie="\n".join(f"    .io_out_rd_0_data_bits_0_{i}(8'sd0)," for i in range(DIM)),
        acc_load="\n".join(f"        acc_d[{i}] <= accm[acc_ib][{i}];" for i in range(DIM)),
        acc_store="\n".join(f"        accm[acc_wi][{i}] <= acc_wd[{i}];" for i in range(DIM)),
    )
    with open(f"{out}/tb.sv", "w") as f:
        f.write(tb)


TB = r"""
`timescale 1ns/1ps
module tb;
  localparam DIM = %(dim)d, NROW = %(nrow)d;
  reg clk = 1'b0, rst = 1'b1;
  always #1 clk = ~clk;

  reg  [31:0] accm [0:NROW-1][0:DIM-1];
  reg  [31:0] want [0:NROW-1][0:DIM-1];
  reg  signed [31:0] acc_d [0:DIM-1];
  wire signed [31:0] acc_wd [0:DIM-1];
  wire signed [7:0]  out_wd [0:DIM-1];
  reg  acc_dv = 0, uop_dv = 0, start = 0;
  reg  [10:0] u0 = 0, u1 = 0; reg [9:0] u2 = 0;
  wire acc_iv, uop_iv, acc_wv, done;
  wire [10:0] acc_ib, uop_ib, acc_wi;
  integer cyc = 0, first_in = -1, last_wr = -1, i, j, errs = 0;

  TensorAlu dut (
    .clock(clk), .reset(rst), .io_start(start), .io_done(done),
    .io_dec_alu_imm(16'd%(shift)d), .io_dec_alu_use_imm(1'b1),
    .io_dec_alu_op(3'd%(op)d),
    .io_dec_src_1(11'd0), .io_dec_src_0(11'd%(dim)d),
    .io_dec_dst_1(11'd0), .io_dec_dst_0(11'd%(dim)d),
    .io_dec_empty_0(1'b0),
    .io_dec_lp_1(14'd1), .io_dec_lp_0(14'd%(nrow_div)d),
    .io_dec_uop_end(14'd%(dim)d), .io_dec_uop_begin(13'd0),
    .io_dec_reset(1'b0),
    .io_dec_push_next(1'b0), .io_dec_push_prev(1'b0),
    .io_dec_pop_next(1'b0), .io_dec_pop_prev(1'b0), .io_dec_op(3'd0),
    .io_uop_idx_valid(uop_iv), .io_uop_idx_bits(uop_ib),
    .io_uop_data_valid(uop_dv),
    .io_uop_data_bits_u2(u2), .io_uop_data_bits_u1(u1), .io_uop_data_bits_u0(u0),
    .io_acc_rd_0_idx_valid(acc_iv), .io_acc_rd_0_idx_bits(acc_ib),
    .io_acc_rd_0_data_valid(acc_dv),
%(acc_rd)s
    .io_acc_wr_0_valid(acc_wv), .io_acc_wr_0_bits_idx(acc_wi),
%(acc_wr)s
    .io_out_rd_0_idx_valid(), .io_out_rd_0_idx_bits(),
    .io_out_rd_0_data_valid(1'b0),
%(out_rd_tie)s
    .io_out_wr_0_valid(), .io_out_wr_0_bits_idx(),
%(out_wr)s
  );

  always @(posedge clk) begin
    uop_dv <= uop_iv;
    if (uop_iv) begin u0 <= uop_ib[10:0]; u1 <= uop_ib[10:0]; u2 <= 10'd0; end
    acc_dv <= acc_iv;
    if (acc_iv) begin
%(acc_load)s
    end
    if (acc_wv) begin
%(acc_store)s
      last_wr = cyc;
    end
    if (!rst) cyc = cyc + 1;
    if (!rst && uop_iv && first_in < 0) first_in = cyc;
  end

  initial begin
    $readmemh("acc.dat", accm);
    $readmemh("want.dat", want);
    repeat (8) @(posedge clk);
    rst = 0; @(posedge clk);
    start = 1; @(posedge clk); start = 0;
    for (i = 0; i < NROW * 200 + 20000; i = i + 1) begin
      @(posedge clk);
      if (done) i = NROW * 200 + 20000;
    end
    repeat (8) @(posedge clk);
    for (i = 0; i < NROW; i = i + 1)
      for (j = 0; j < DIM; j = j + 1)
        if (accm[i][j] !== want[i][j]) begin
          if (errs < 8)
            $display("VTAALU MISMATCH acc[%%0d][%%0d] got=%%0d want=%%0d",
                     i, j, $signed(accm[i][j]), $signed(want[i][j]));
          errs = errs + 1;
        end
    $display("VTAALU RESULT rows=%%0d errs=%%0d cycles=%%0d first_in=%%0d last_wr=%%0d",
             NROW, errs, (last_wr - first_in), first_in, last_wr);
    $display("VTAALU %%s", (errs == 0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--shift", type=int, default=8)
    ap.add_argument("--rtl",
                    default="/scratch/hc676/vta/hardware/chisel/vta_out_TensorAlu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.rows % DIM:
        sys.exit(f"rows must be a multiple of {DIM}")
    global TB
    TB = TB.replace("%(nrow_div)d", str(a.rows // DIM))
    emit(a.rows, a.shift, a.out)
    print(f"VTAALU BENCH rows={a.rows} shift={a.shift} -> {a.out}")
    if not a.run:
        return
    v = [os.path.join(a.rtl, f) for f in sorted(os.listdir(a.rtl))
         if f.endswith(".v")]
    for cmd in (["xvlog"] + v, ["xvlog", "-sv", "tb.sv"],
                ["xelab", "tb", "-s", "tbsim", "-timescale", "1ns/1ps",
                 "-L", "unisims_ver", "-L", "unimacro_ver", "-L", "secureip"],
                ["xsim", "tbsim", "-runall"]):
        r = subprocess.run(cmd, cwd=a.out, capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()
        if r.returncode:
            print(f"FAILED: {' '.join(cmd[:2])}")
            print("\n".join(tail[-22:]))
            sys.exit(1)
        for line in tail:
            if line.startswith("VTAALU ") or "Error" in line:
                print(line)


if __name__ == "__main__":
    main()
