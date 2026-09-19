#!/usr/bin/env python3
"""An xsim bench for Gemmini's normalise/scale path, driven beat by beat.

The schedule is built in Python and played by the testbench, so the beat
sequence -- which `NormCmd` on which beat, which statistics bank, when the
output pass starts -- is written where it can be read, and the Verilog is a
player with no policy in it.

The sequence itself is not a choice.  `Normalizer`'s state machine fixes it:

  `io.in.fire` sets the state to `output` for `RESET`, `get_max` for `MAX`,
  and `get_sum` for everything else, and each beat carries `len` elements
  which `count` accumulates.  From `get_sum`, `SUM`/`VARIANCE`/`SUM_EXP`
  return to idle -- they only accumulate -- while `MEAN` runs the divider,
  `INV_STDDEV` runs divider then square root then reciprocal, and
  `INV_SUM_EXP` runs the 127-over-sum divide.  So a row is: `n-1` beats of
  the accumulating command and one of the finishing command, per pass.

Three passes over a row, which is why the SPMW side reads its accumulator
three times: this is the discipline being matched, not an implementation
detail of one side.
"""

import argparse
import os
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.environ.get("SPMW_TESTS",
                                  "/scratch/hc676/allo/tests/dataflow/spmw"))
from spmw_block_drive import (  # noqa: E402
    IEXP_QB, IEXP_QC, IEXP_QLN2, IEXP_QLN2I, IGELU_QB, IGELU_QC, MODE_NAME,
    NAME_MODE, SCALE,
)
from spmw_block_engine import M_GELU, M_LN, M_NONE, M_RELU, M_SM  # noqa: E402
from spmw_block_ref import norm_path  # noqa: E402

DIM = 16
#: `NormCmd` as a ChiselEnum, in declaration order.
C_RESET, C_SUM, C_MEAN, C_VARIANCE, C_INV_STDDEV, C_MAX, C_SUM_EXP, \
    C_INV_SUM_EXP = range(8)


def schedule(mode, nrow, nbeat, banks=2):
    """`(row, beat, cmd, act, stats_id, is_output)` for every input beat."""
    if mode == M_LN:
        passes = [([C_SUM] * (nbeat - 1) + [C_MEAN], 0),
                  ([C_VARIANCE] * (nbeat - 1) + [C_INV_STDDEV], 0),
                  ([C_RESET] * nbeat, M_LN)]
    elif mode == M_SM:
        passes = [([C_MAX] * nbeat, 0),
                  ([C_SUM_EXP] * (nbeat - 1) + [C_INV_SUM_EXP], 0),
                  ([C_RESET] * nbeat, M_SM)]
    else:
        passes = [([C_RESET] * nbeat, mode)]
    out = []
    for r in range(nrow):
        for cmds, act in passes:
            for b, c in enumerate(cmds):
                out.append((r, b, c, act, r % banks, c == C_RESET))
    return out


TB = r"""
`timescale 1ns/1ps
module tb;
  localparam DIM = %(dim)d;
  localparam NBEAT = %(nbeat)d;
  localparam NSCHED = %(nsched)d;
  localparam NOUT = %(nout)d;

  reg clock = 1'b0, reset = 1'b1;
  always #1 clock = ~clock;

  reg  [31:0] acc   [0:%(nacc_m1)d][0:DIM-1];   // the accumulator rows
  reg  [31:0] want  [0:NOUT-1][0:DIM-1];        // the golden int8, sign-extended
  reg  [31:0] sched [0:NSCHED-1][0:4];          // beat, cmd, act, stats_id, is_out

  reg         acc_valid = 1'b0;
  reg  [31:0] acc_bits [0:DIM-1];
  reg  [2:0]  r_act = 3'd0, r_cmd = 3'd0;
  reg         r_sid = 1'b0;
  wire        acc_ready, out_valid;
  wire [7:0]  out_data [0:DIM-1];

  integer si = 0, oi = 0, cyc = 0, errs = 0, first_in = -1, last_out = -1;
  integer i;

  MxuVpuNorm dut (
    .clock(clock), .reset(reset),
    .io_a_valid(1'b0), .io_b_valid(1'b0), .io_d_valid(1'b0),
    .io_req_valid(1'b0),
    .io_req_bits_tag_id(8'd0), .io_req_bits_pe_control_dataflow(1'b0),
    .io_req_bits_pe_control_propagate(1'b0), .io_req_bits_pe_control_shift(5'd0),
    .io_req_bits_a_transpose(1'b0), .io_req_bits_bd_transpose(1'b0),
    .io_req_bits_total_rows(5'd%(dim)d), .io_req_bits_flush(2'd0),
%(a_ties)s
    .io_acc_in_valid(acc_valid), .io_acc_in_ready(acc_ready),
%(acc_conn)s
    .io_scale_bits(32'h%(scale)08x), .io_act(r_act), .io_cmd(r_cmd),
    .io_len(11'd%(dim)d), .io_stats_id(r_sid),
    .io_igelu_qb(32'd%(qb)d), .io_igelu_qc(32'd%(qc)d),
    .io_iexp_qln2(32'd%(qln2)d), .io_iexp_qln2_inv(32'd%(qln2i)d),
    .io_out_ready(1'b1), .io_out_valid(out_valid),
%(out_conn)s
  );

  // Drive: hold `valid` and advance on `ready`, so the count is the design's
  // own rate and not the bench's.
  always @(posedge clock) begin
    if (reset) begin
      acc_valid <= 1'b0;
    end else begin
      if (!acc_valid || acc_ready) begin
        if (si < NSCHED) begin
          for (i = 0; i < DIM; i = i + 1)
            acc_bits[i] <= acc[sched[si][0]][i];
          r_cmd <= sched[si][1][2:0];
          r_act <= sched[si][2][2:0];
          r_sid <= sched[si][3][0];
          acc_valid <= 1'b1;
          if (first_in < 0) first_in = cyc;
          si = si + 1;
        end else begin
          acc_valid <= 1'b0;
        end
      end
    end
  end

  // Collect
  always @(posedge clock) begin
    if (!reset) begin
      cyc = cyc + 1;
      if (out_valid) begin
        last_out = cyc;
        if (oi < NOUT) begin
          for (i = 0; i < DIM; i = i + 1)
            if ($signed({{24{out_data[i][7]}}, out_data[i]}) !== $signed(want[oi][i])) begin
              if (errs < 12)
                $display("GEM MISMATCH out=%%0d lane=%%0d got=%%0d want=%%0d",
                         oi, i, $signed({{24{out_data[i][7]}}, out_data[i]}),
                         $signed(want[oi][i]));
              errs = errs + 1;
            end
        end
        oi = oi + 1;
      end
    end
  end

  initial begin
    $readmemh("acc.dat", acc);
    $readmemh("want.dat", want);
    $readmemh("sched.dat", sched);
    repeat (8) @(posedge clock);
    reset = 1'b0;
    for (i = 0; i < %(timeout)d; i = i + 1) begin
      @(posedge clock);
      if (oi >= NOUT) i = %(timeout)d;
    end
    $display("GEM RESULT mode=%(mode)s outs=%%0d/%%0d errs=%%0d cycles=%%0d first_in=%%0d last_out=%%0d",
             oi, NOUT, errs, (last_out - first_in), first_in, last_out);
    $display("GEM %%s", (oi == NOUT && errs == 0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
"""


def build(mode, nrow, ln, out, seed=0):
    nbeat = ln // DIM
    nacc = nrow * nbeat
    rng = np.random.default_rng(seed + 7)
    acc = rng.integers(-3000, 3000, size=(nrow, ln)).astype(np.int64)
    qb, qc = (IGELU_QB, IGELU_QC) if mode == M_GELU else (IEXP_QB, IEXP_QC)
    want, _, _ = norm_path(acc, mode, SCALE[mode], qb, qc, IEXP_QLN2,
                           IEXP_QLN2I)
    want = want.reshape(nacc, DIM)
    sched = schedule(mode, nrow, nbeat)

    os.makedirs(out, exist_ok=True)
    flat = acc.reshape(nacc, DIM).astype(np.int64)
    with open(f"{out}/acc.dat", "w") as f:
        for row in flat:
            f.write(" ".join(f"{v & 0xFFFFFFFF:08x}" for v in row) + "\n")
    with open(f"{out}/want.dat", "w") as f:
        for row in want.astype(np.int64):
            f.write(" ".join(f"{v & 0xFFFFFFFF:08x}" for v in row) + "\n")
    with open(f"{out}/sched.dat", "w") as f:
        for r, b, c, a, sid, _ in sched:
            f.write(f"{r * nbeat + b:08x} {c:08x} {a:08x} {sid:08x} 00000000\n")

    scale_bits = struct.unpack("<I", struct.pack("<f", SCALE[mode]))[0]
    tb = TB % dict(
        dim=DIM, nbeat=nbeat, nsched=len(sched), nout=nacc, nacc_m1=nacc - 1,
        a_ties="\n".join(
            f"    .io_{p}_bits_{i}_0(8'd0)," for p in ("a", "b", "d")
            for i in range(DIM)),
        acc_conn="\n".join(
            f"    .io_acc_in_bits_{i}_0(acc_bits[{i}])," for i in range(DIM)),
        out_conn=",\n".join(
            f"    .io_out_bits_data_{i}_0(out_data[{i}])" for i in range(DIM)),
        scale=scale_bits, qb=qb & 0xFFFFFFFF, qc=qc & 0xFFFFFFFF,
        qln2=IEXP_QLN2, qln2i=IEXP_QLN2I,
        timeout=200 * nacc + 20000, mode=MODE_NAME[mode],
    )
    with open(f"{out}/tb.sv", "w") as f:
        f.write(tb)
    return len(sched), nacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="layernorm")
    ap.add_argument("--nrow", type=int, default=4)
    ap.add_argument("--len", type=int, default=64, dest="ln")
    ap.add_argument("--rtl",
                    default="/scratch/hc676/e3_micro/gemmini/mxuvpunorm_out_16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    mode = NAME_MODE[a.mode]
    nsched, nout = build(mode, a.nrow, a.ln, a.out)
    print(f"GEM BENCH mode={a.mode} nrow={a.nrow} len={a.ln} "
          f"beats={nsched} outputs={nout} -> {a.out}")
    if not a.run:
        return
    v = [os.path.join(a.rtl, f) for f in sorted(os.listdir(a.rtl))
         if f.endswith(".v")]
    for cmd in (["xvlog"] + v, ["xvlog", "-sv", "tb.sv"],
                # `-timescale`: the Chisel output carries no timescale
                # directive and the bench does, which xelab rejects outright
                # rather than defaulting.
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
            if line.startswith("GEM ") or "Error" in line:
                print(line)


if __name__ == "__main__":
    main()
