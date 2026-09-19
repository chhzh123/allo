#!/usr/bin/env python3
"""Gemmini's mesh, measured on `MxuVpuNorm` itself rather than carried over.

E3 measured `MeshWithDelays` at 16x16 through `MxuVpu` and got an interval of
`S + 2`.  The mesh here is the same module at the same width, so the number
ought to carry -- but "ought to" is what this bench replaces.  It drives the
mesh directly and reads `io_mesh_out`, which is the raw int32 result before
the scale path, so the measurement is the mesh and nothing else.

The protocol is E3's, from `e3_tpu/micro/scripts/gen_mxuvpu_tb.py`: one `req`
a tile carrying the tag and the propagate bit, `d` shifting the next tile's
weights in bottom row first while `a` streams this tile's activations, and a
warm-up pass because the weights are double-buffered -- sixteen tiles take
seventeen passes.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

DIM = 16

TB = r"""
`timescale 1ns/1ps
module tb;
  localparam DIM = %(dim)d;
  localparam TILES = %(tiles)d;

  reg clk = 1'b0, rst = 1'b1;
  always #1 clk = ~clk;

  reg signed [7:0] A [0:TILES*DIM*DIM-1];
  reg signed [7:0] B [0:TILES*DIM*DIM-1];
  reg signed [31:0] WANT [0:TILES*DIM*DIM-1];

  reg signed [7:0] a_bits [0:DIM-1];
  reg signed [7:0] d_bits [0:DIM-1];
  reg a_valid = 0, d_valid = 0, b_valid = 0, req_valid = 0;
  wire a_ready, d_ready, b_ready, req_ready;
  wire out_valid;
  wire signed [31:0] out_bits [0:DIM-1];

  integer cyc = 0;
  always @(posedge clk) if (!rst) cyc = cyc + 1;

  MxuVpuNorm dut (
    .clock(clk), .reset(rst),
    .io_a_ready(a_ready), .io_a_valid(a_valid),
%(a_conn)s
    .io_b_ready(b_ready), .io_b_valid(b_valid),
%(b_conn)s
    .io_d_ready(d_ready), .io_d_valid(d_valid),
%(d_conn)s
    .io_req_ready(req_ready), .io_req_valid(req_valid),
    .io_req_bits_tag_id(8'd1),
    .io_req_bits_pe_control_dataflow(1'b1),
    .io_req_bits_pe_control_propagate(1'b1),
    .io_req_bits_pe_control_shift(5'd0),
    .io_req_bits_a_transpose(1'b0), .io_req_bits_bd_transpose(1'b0),
    .io_req_bits_total_rows(5'd%(dim)d), .io_req_bits_flush(2'd0),
    .io_acc_in_valid(1'b0),
%(acc_tie)s
    .io_mesh_out_valid(out_valid),
%(mesh_conn)s
    .io_scale_bits(32'h3b800000), .io_act(3'd0), .io_cmd(3'd0),
    .io_len(11'd%(dim)d), .io_stats_id(1'b0),
    .io_igelu_qb(32'd0), .io_igelu_qc(32'd0),
    .io_iexp_qln2(32'd0), .io_iexp_qln2_inv(32'd0),
    .io_out_ready(1'b1)
  );

  integer rows = 0, first_in = -1, last_out = -1, wrong = 0, i;
  integer tile_done [0:TILES-1];

  always @(posedge clk) if (!rst) begin
    if (out_valid) begin
      // Pass 0 is the warm-up and carries no activations; after it every DIM
      // rows completes one tile.
      if (rows >= DIM && (rows - DIM) < TILES*DIM)
        for (i = 0; i < DIM; i = i + 1)
          if (out_bits[i] !== WANT[(rows-DIM)*DIM + i]) begin
            if (wrong < 8)
              $display("MESH MISMATCH row=%%0d lane=%%0d got=%%0d want=%%0d",
                       rows-DIM, i, out_bits[i], WANT[(rows-DIM)*DIM+i]);
            wrong = wrong + 1;
          end
      rows = rows + 1;
      last_out = cyc;
      if (rows %% DIM == 0 && rows/DIM >= 2 && rows/DIM - 2 < TILES)
        if (tile_done[rows/DIM - 2] < 0) tile_done[rows/DIM - 2] = cyc;
    end
    if (a_valid && a_ready && first_in < 0) first_in = cyc;
  end

  task automatic issue();
    integer guard;
    begin
      req_valid = 1; guard = 0;
      while (req_ready !== 1'b1 && guard < 100) begin @(posedge clk); guard = guard + 1; end
      @(posedge clk); req_valid = 0;
    end
  endtask

  task automatic pass(input integer act, input integer wt);
    integer row, guard, c;
    begin
      row = 0; guard = 0;
      while (row < DIM && guard < 500) begin
        a_valid = 1; b_valid = 1; d_valid = 1;
        for (c = 0; c < DIM; c = c + 1) begin
          a_bits[c] = (act < 0) ? 8'sd0 : A[act*DIM*DIM + row*DIM + c];
          d_bits[c] = (wt  < 0) ? 8'sd0 : B[wt*DIM*DIM + (DIM-1-row)*DIM + c];
        end
        @(posedge clk);
        if (a_ready === 1'b1) row = row + 1;
        guard = guard + 1;
      end
      a_valid = 0; b_valid = 0; d_valid = 0;
    end
  endtask

  integer t, gmin, gmax, g;
  initial begin
    $readmemh("a.dat", A); $readmemh("b.dat", B); $readmemh("want.dat", WANT);
    for (t = 0; t < TILES; t = t + 1) tile_done[t] = -1;
    repeat (8) @(posedge clk);
    rst = 0; @(posedge clk);
    issue(); pass(-1, 0);
    for (t = 0; t < TILES; t = t + 1) begin
      issue();
      pass(t, (t + 1 < TILES) ? t + 1 : -1);
    end
    for (t = 0; t < 4000 && rows < (TILES+1)*DIM; t = t + 1) @(posedge clk);
    gmin = 1000000; gmax = -1;
    for (t = 1; t < TILES; t = t + 1) begin
      g = tile_done[t] - tile_done[t-1];
      if (g < gmin) gmin = g;
      if (g > gmax) gmax = g;
    end
    $display("MESH RESULT dim=%%0d tiles=%%0d rows=%%0d wrong=%%0d latency=%%0d interval_min=%%0d interval_max=%%0d total=%%0d",
             DIM, TILES, rows, wrong, tile_done[0]-first_in, gmin, gmax, last_out-first_in);
    $display("MESH %%s", (wrong == 0 && rows >= (TILES+1)*DIM) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=16)
    ap.add_argument("--rtl",
                    default="/scratch/hc676/e3_micro/gemmini/mxuvpunorm_out_16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(5)
    A = rng.integers(-8, 8, size=(a.tiles, DIM, DIM)).astype(np.int64)
    B = rng.integers(-8, 8, size=(a.tiles, DIM, DIM)).astype(np.int64)
    want = np.stack([A[t] @ B[t] for t in range(a.tiles)])

    os.makedirs(a.out, exist_ok=True)
    for name, arr in (("a", A), ("b", B), ("want", want)):
        with open(f"{a.out}/{name}.dat", "w") as f:
            for v in arr.ravel():
                f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")

    tb = TB % dict(
        dim=DIM, tiles=a.tiles,
        a_conn="\n".join(f"    .io_a_bits_{i}_0(a_bits[{i}])," for i in range(DIM)),
        b_conn="\n".join(f"    .io_b_bits_{i}_0(8'sd0)," for i in range(DIM)),
        d_conn="\n".join(f"    .io_d_bits_{i}_0(d_bits[{i}])," for i in range(DIM)),
        acc_tie="\n".join(f"    .io_acc_in_bits_{i}_0(32'd0)," for i in range(DIM)),
        mesh_conn="\n".join(
            f"    .io_mesh_out_bits_{i}_0(out_bits[{i}])," for i in range(DIM)),
    )
    with open(f"{a.out}/tb.sv", "w") as f:
        f.write(tb)
    print(f"MESH BENCH dim={DIM} tiles={a.tiles} -> {a.out}")
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
            print("\n".join(tail[-20:]))
            sys.exit(1)
        for line in tail:
            if line.startswith("MESH ") or "Error" in line:
                print(line)


if __name__ == "__main__":
    main()
