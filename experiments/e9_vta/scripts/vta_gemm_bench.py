#!/usr/bin/env python3
"""An xsim bench for VTA's GEMM core, on the same footing as Gemmini's.

Every data port carries a leading batch index (`..._data_bits_0_<i>`)
because `batch = 1`; only the weight port is two-dimensional over the matrix.

`TensorGemm` is a controller over four scratchpads it does not own -- micro-op,
input, weight and accumulator -- so the bench models all four.  The weight port
is 256 int8 wide because VTA reads the **entire 16x16 matrix every cycle**: it
is not weight-stationary, where Gemmini and SPMW hold one weight per cell and
stream activations past it.  That is the architectural difference this bench
makes visible, and it is why the port list is 657 signals.

The instruction is three nested loops -- `lp_0`, `lp_1`, and `uop_begin` to
`uop_end` -- and each innermost step issues one matrix-vector multiply of
1x16 by 16x16.  One 16x16x16 tile is therefore 16 issues, and `T` tiles is
`lp_0 = T` with sixteen micro-ops.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

DIM = 16


def emit(nt, out):
    """Write the bench and its stimulus for `nt` back-to-back tiles."""
    rng = np.random.default_rng(11)
    A = rng.integers(-8, 8, size=(nt, DIM, DIM)).astype(np.int64)   # activations
    W = rng.integers(-8, 8, size=(nt, DIM, DIM)).astype(np.int64)   # weights
    # One micro-op a row: accumulator k, input row k, weight matrix 0 (+ the
    # per-loop offsets), so a tile is out[r][:] = A[t][r][:] @ W[t].
    want = np.stack([A[t] @ W[t] for t in range(nt)])

    os.makedirs(out, exist_ok=True)
    with open(f"{out}/inp.dat", "w") as f:            # nt*DIM rows of DIM int8
        for t in range(nt):
            for r in range(DIM):
                f.write(" ".join(f"{v & 0xFF:02x}" for v in A[t][r]) + "\n")
    with open(f"{out}/wgt.dat", "w") as f:            # nt rows of DIM*DIM int8
        for t in range(nt):
            # wgt_d[i][j] is used as out[i] += inp[j] * wgt[i][j], so the
            # matrix is stored transposed relative to `A @ W`.
            f.write(" ".join(f"{int(W[t][j][i]) & 0xFF:02x}"
                             for i in range(DIM) for j in range(DIM)) + "\n")
    with open(f"{out}/want.dat", "w") as f:           # nt*DIM rows of DIM int32
        for t in range(nt):
            for r in range(DIM):
                f.write(" ".join(f"{int(v) & 0xFFFFFFFF:08x}"
                                 for v in want[t][r]) + "\n")

    n = lambda i: f"{i}"
    tb = TB % dict(
        dim=DIM, nt=nt, nacc=nt * DIM, ninp=nt * DIM, nwgt=nt,
        inp_rd="\n".join(f"    .io_inp_rd_0_data_bits_0_{i}(inp_d[{i}])," for i in range(DIM)),
        wgt_rd="\n".join(f"    .io_wgt_rd_0_data_bits_{i}_{j}(wgt_d[{i}][{j}]),"
                         for i in range(DIM) for j in range(DIM)),
        acc_rd="\n".join(f"    .io_acc_rd_0_data_bits_0_{i}(acc_d[{i}])," for i in range(DIM)),
        acc_wr="\n".join(f"    .io_acc_wr_0_bits_data_0_{i}(acc_wd[{i}])," for i in range(DIM)),
        out_wr="\n".join(f"    .io_out_wr_0_bits_data_0_{i}(out_wd[{i}])," for i in range(DIM)),
        inp_wr_tie="\n".join(f"    .io_inp_wr_0_bits_data_0_{i}()," for i in range(DIM)),
        wgt_wr_tie="\n".join(f"    .io_wgt_wr_0_bits_data_{i}_{j}(),"
                             for i in range(DIM) for j in range(DIM)),
        out_rd_tie="\n".join(f"    .io_out_rd_0_data_bits_0_{i}(8'sd0)," for i in range(DIM)),
        inp_load="\n".join(f"        inp_d[{i}] <= inpm[inp_ib][{i}];" for i in range(DIM)),
        wgt_load="\n".join(f"        wgt_d[{i}][{j}] <= wgtm[wgt_ib][{i}][{j}];"
                           for i in range(DIM) for j in range(DIM)),
        acc_load="\n".join(f"        acc_d[{i}] <= accm[acc_ib][{i}];" for i in range(DIM)),
        acc_store="\n".join(f"        accm[acc_wi][{i}] <= acc_wd[{i}];" for i in range(DIM)),
    )
    with open(f"{out}/tb.sv", "w") as f:
        f.write(tb)
    return nt * DIM


TB = r"""
`timescale 1ns/1ps
module tb;
  localparam DIM = %(dim)d, NT = %(nt)d, NACC = %(nacc)d, NINP = %(ninp)d, NWGT = %(nwgt)d;

  reg clk = 1'b0, rst = 1'b1;
  always #1 clk = ~clk;

  // --- the four scratchpads TensorGemm reads but does not own --------------
  reg  [31:0] uopm  [0:DIM-1];                      // u0 | u1<<16 packed
  reg  [7:0]  inpm  [0:NINP-1][0:DIM-1];
  reg  [7:0]  wgtm  [0:NWGT-1][0:DIM-1][0:DIM-1];
  reg  [31:0] accm  [0:NACC-1][0:DIM-1];
  reg  [31:0] want  [0:NACC-1][0:DIM-1];

  reg  signed [7:0]  inp_d [0:DIM-1];
  reg  signed [7:0]  wgt_d [0:DIM-1][0:DIM-1];
  reg  signed [31:0] acc_d [0:DIM-1];
  wire signed [31:0] acc_wd [0:DIM-1];
  wire signed [7:0]  out_wd [0:DIM-1];
  reg  inp_dv = 0, wgt_dv = 0, acc_dv = 0, uop_dv = 0;
  reg  [10:0] u0 = 0, u1 = 0; reg [9:0] u2 = 0;

  wire inp_iv, wgt_iv, acc_iv, uop_iv, acc_wv, out_wv, done;
  wire [10:0] inp_ib, acc_ib, uop_ib, acc_wi;
  wire [9:0]  wgt_ib;
  reg  start = 0;
  integer cyc = 0, first_in = -1, last_wr = -1, i, j, t, r, errs = 0;

  TensorGemm dut (
    .clock(clk), .reset(rst),
    .io_start(start), .io_done(done),
    .io_dec_wgt_1(10'd0), .io_dec_wgt_0(10'd1),
    .io_dec_inp_1(11'd0), .io_dec_inp_0(11'd%(dim)d),
    .io_dec_acc_1(11'd0), .io_dec_acc_0(11'd%(dim)d),
    .io_dec_empty_0(1'b0),
    .io_dec_lp_1(14'd1), .io_dec_lp_0(14'd%(nt)d),
    .io_dec_uop_end(14'd%(dim)d), .io_dec_uop_begin(13'd0),
    .io_dec_reset(1'b0),
    .io_dec_push_next(1'b0), .io_dec_push_prev(1'b0),
    .io_dec_pop_next(1'b0), .io_dec_pop_prev(1'b0), .io_dec_op(3'd0),
    .io_uop_idx_valid(uop_iv), .io_uop_idx_bits(uop_ib),
    .io_uop_data_valid(uop_dv),
    .io_uop_data_bits_u2(u2), .io_uop_data_bits_u1(u1), .io_uop_data_bits_u0(u0),
    .io_inp_rd_0_idx_valid(inp_iv), .io_inp_rd_0_idx_bits(inp_ib),
    .io_inp_rd_0_data_valid(inp_dv),
%(inp_rd)s
    .io_inp_wr_0_valid(), .io_inp_wr_0_bits_idx(),
%(inp_wr_tie)s
    .io_wgt_rd_0_idx_valid(wgt_iv), .io_wgt_rd_0_idx_bits(wgt_ib),
    .io_wgt_rd_0_data_valid(wgt_dv),
%(wgt_rd)s
    .io_wgt_wr_0_valid(), .io_wgt_wr_0_bits_idx(),
%(wgt_wr_tie)s
    .io_acc_rd_0_idx_valid(acc_iv), .io_acc_rd_0_idx_bits(acc_ib),
    .io_acc_rd_0_data_valid(acc_dv),
%(acc_rd)s
    .io_acc_wr_0_valid(acc_wv), .io_acc_wr_0_bits_idx(acc_wi),
%(acc_wr)s
    .io_out_rd_0_idx_valid(), .io_out_rd_0_idx_bits(),
    .io_out_rd_0_data_valid(1'b0),
%(out_rd_tie)s
    .io_out_wr_0_valid(out_wv), .io_out_wr_0_bits_idx(),
%(out_wr)s
    .io_state(), .io_inflight()
  );

  // one-cycle synchronous read, which is what VTA's own scratchpads give
  always @(posedge clk) begin
    uop_dv <= uop_iv;
    if (uop_iv) begin u0 <= uopm[uop_ib][10:0]; u1 <= uopm[uop_ib][26:16]; u2 <= 10'd0; end
    inp_dv <= inp_iv;
    if (inp_iv) begin
%(inp_load)s
    end
    wgt_dv <= wgt_iv;
    if (wgt_iv) begin
%(wgt_load)s
    end
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
    $readmemh("inp.dat", inpm);
    $readmemh("wgt.dat", wgtm);
    $readmemh("want.dat", want);
    for (i = 0; i < DIM; i = i + 1) uopm[i] = (i << 16) | i;   // u1=row, u0=acc
    for (i = 0; i < NACC; i = i + 1)
      for (j = 0; j < DIM; j = j + 1) accm[i][j] = 32'd0;
    repeat (8) @(posedge clk);
    rst = 0; @(posedge clk);
    start = 1; @(posedge clk); start = 0;
    for (i = 0; i < %(nt)d * 400 + 20000; i = i + 1) begin
      @(posedge clk);
      if (done) i = %(nt)d * 400 + 20000;
    end
    repeat (8) @(posedge clk);
    for (i = 0; i < NACC; i = i + 1)
      for (j = 0; j < DIM; j = j + 1)
        if (accm[i][j] !== want[i][j]) begin
          if (errs < 8)
            $display("VTA MISMATCH acc[%%0d][%%0d] got=%%0d want=%%0d",
                     i, j, $signed(accm[i][j]), $signed(want[i][j]));
          errs = errs + 1;
        end
    $display("VTA RESULT tiles=%%0d errs=%%0d cycles=%%0d first_in=%%0d last_wr=%%0d per_tile=%%0d",
             NT, errs, (last_wr - first_in), first_in, last_wr,
             (last_wr - first_in) / NT);
    $display("VTA %%s", (errs == 0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=16)
    ap.add_argument("--rtl",
                    default="/scratch/hc676/vta/hardware/chisel/vta_out_TensorGemm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    n = emit(a.tiles, a.out)
    print(f"VTA BENCH tiles={a.tiles} acc_rows={n} -> {a.out}")
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
            print("\n".join(tail[-25:]))
            sys.exit(1)
        for line in tail:
            if line.startswith("VTA ") or "Error" in line:
                print(line)


if __name__ == "__main__":
    main()
