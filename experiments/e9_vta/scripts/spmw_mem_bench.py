#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An xsim bench for SPMW's whole engine, running a program from memory.

`scripts/spmw_build_array.py --cosim` checks the engine of
`test_spmw_ptpu_mem` against streams: the beats memory would answer with are
there before they are asked for, so its cycle count has no memory in it. This
bench puts a memory behind the engine's two ports, as `vta_core_bench.py` does
behind VTA's `Core` and `gemmini_rocc_bench.py` behind Gemmini, and counts
from the launch to `done`:

* the read port takes a request whenever it has room and returns a 64-bit
  beat every cycle from the next, the requests answered in order;
* the write port takes a request when it is idle, then that request's beats,
  and acknowledges the cycle after the last of them.

That is the memory VTA's bench has, a read channel of it and its write
channel, so the two engines see the same one. ``--stall P`` makes it a worse
one: every cycle each of its four handshakes is withheld with probability
``P`` percent, which is what shows whether the array survives waiting.

The program is one GEMM instruction and its operands, written from the same
stimulus file the other two engines' benches read, or with ``--mixed`` the
five GEMMs of `test_spmw_ptpu_mem`. What the engine leaves in memory is
compared with the golden result, word for word, over the whole image.

The hardware is an array build's: this compiles the `sim/` directory a
`--cosim` build leaves behind against its own testbench, so one build of a
size runs every workload.

Reported, in cycles from the launch token to the `done` token:

    total     end to end
    floor     the launch's rows: one a cycle is the array's best
    rows      tokens the head put on the array's edge: the floor and the
              ``S`` beats that carry the first weights in
    first     cycles from the launch to the first of them
    paused    cycles between the first and the last in which it put none
    rd_beats  beats the read port carried, and rd_reqs its requests
    wr_beats  beats the write port carried, and wr_reqs its requests
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tests", "dataflow", "spmw"))

# pylint: disable=wrong-import-position
from test_spmw_ptpu_mem import Gemm, Launch, mixed_gemms  # noqa: E402

#: The boundary streams of `spmw_top`, by the port each one ends on.
STREAMS = ("launch", "rd_cmd", "rd_data", "wr_ack", "wr_cmd", "wr_data", "done")


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


def program(stim_path, S):
    """The stimulus as one GEMM instruction, and the launch that runs it."""
    stim = read_stim(stim_path)
    bias = stim["bias"]
    weights = stim["B"] if stim["tiles"] > 1 else stim["B"][0]
    gemm = Gemm(stim["A"], weights, stim["shift"], bias=bias, relu=bias is not None)
    if not np.array_equal(gemm.golden(), stim["C"]):
        sys.exit("the instruction's result is not the stimulus file's golden one")
    return stim, Launch(S, [gemm])


def hex_words(image):
    """A byte image as 64-bit little-endian words, one a line."""
    return "\n".join(f"{int(w):016x}" for w in image.view("<u8")) + "\n"


def boundary(top):
    """Each boundary stream's family name, read off `spmw_top`'s ports."""
    text = open(top, encoding="utf-8").read()
    names = {}
    for stream in STREAMS:
        found = re.search(rf"\b(\w+_{stream}_bind)_(?:dout|din)\b", text)
        if not found:
            sys.exit(f"{top} has no boundary stream ending on `{stream}`")
        names[stream] = found.group(1)
    return names


def testbench(names, words, limit, stall=0):
    """The bench: a memory behind the two ports, and the count."""

    def inbound(stream, sig):
        name = names[stream]
        return (
            f"    .{name}_dout({sig}_dout), .{name}_empty_n({sig}_empty_n), "
            f".{name}_read({sig}_read),"
        )

    def outbound(stream, sig):
        name = names[stream]
        return (
            f"    .{name}_din({sig}_din), .{name}_write({sig}_write), "
            f".{name}_full_n({sig}_full_n),"
        )

    ports = "\n".join(
        [
            inbound("launch", "la"),
            outbound("rd_cmd", "rc"),
            inbound("rd_data", "rd"),
            inbound("wr_ack", "wa"),
            outbound("wr_cmd", "wc"),
            outbound("wr_data", "wd"),
            outbound("done", "dn"),
        ]
    ).rstrip(",")
    return f"""// Generated by spmw_mem_bench.py -- do not edit.
`timescale 1ns/1ps

module tbm;
  localparam int WORDS = {words};
  localparam int QD = 64;
  localparam int STALL = {stall};
  reg clk = 0, rst_n = 0;
  always #5 clk = ~clk;

  // a memory that is not always ready: each handshake withheld STALL% of cycles
  reg rq_go = 1, rd_go = 1, wq_go = 1, wd_go = 1, wa_go = 1;
  always @(posedge clk) begin
    rq_go <= $urandom_range(99) >= STALL;
    rd_go <= $urandom_range(99) >= STALL;
    wq_go <= $urandom_range(99) >= STALL;
    wd_go <= $urandom_range(99) >= STALL;
    wa_go <= $urandom_range(99) >= STALL;
  end

  reg [63:0] mem [0:WORDS-1];
  reg [63:0] want [0:WORDS-1];

  // the launch: one token, the program's address
  reg         launch = 0;
  wire [63:0] la_dout [0:0];
  wire        la_empty_n [0:0], la_read [0:0];
  assign la_dout[0] = 64'd0;
  assign la_empty_n[0] = launch;

  // the read port: a request whenever there is room, a beat a cycle from the next
  wire [63:0] rc_din [0:0], rd_dout [0:0];
  wire        rc_write [0:0], rc_full_n [0:0], rd_empty_n [0:0], rd_read [0:0];
  reg  [31:0] r_qaddr [0:QD-1];
  reg  [31:0] r_qlen [0:QD-1];
  integer r_head = 0, r_tail = 0, r_beat = 0, r_beats = 0, r_reqs = 0;
  assign rc_full_n[0] = ((r_tail - r_head) < QD) && rq_go;
  assign rd_empty_n[0] = (r_head != r_tail) && rd_go;
  assign rd_dout[0] = mem[(r_qaddr[r_head % QD] >> 3) + r_beat];
  always @(posedge clk) if (rst_n) begin
    if (rc_write[0] && rc_full_n[0]) begin
      r_qaddr[r_tail % QD] <= rc_din[0][31:0];
      r_qlen[r_tail % QD] <= rc_din[0][63:32];
      r_tail <= r_tail + 1;
      r_reqs <= r_reqs + 1;
    end
    if (rd_empty_n[0] && rd_read[0]) begin
      r_beats <= r_beats + 1;
      if (r_beat + 1 == r_qlen[r_head % QD]) begin r_beat <= 0; r_head <= r_head + 1; end
      else r_beat <= r_beat + 1;
    end
  end

  // the write port: a request when idle, its beats, and an acknowledgement
  // the cycle after the last
  wire [63:0] wc_din [0:0], wd_din [0:0], wa_dout [0:0];
  wire        wc_write [0:0], wc_full_n [0:0], wd_write [0:0], wd_full_n [0:0];
  wire        wa_empty_n [0:0], wa_read [0:0];
  reg         w_busy = 0;
  reg  [31:0] w_base = 0, w_left = 0;
  integer w_beat = 0, w_beats = 0, w_reqs = 0, acks = 0;
  assign wc_full_n[0] = !w_busy && wq_go;
  assign wd_full_n[0] = w_busy && wd_go;
  assign wa_empty_n[0] = (acks != 0) && wa_go;
  assign wa_dout[0] = 64'd1;
  wire w_take = wd_write[0] && wd_full_n[0];
  wire w_last = w_take && (w_beat + 1 == w_left);
  always @(posedge clk) if (rst_n) begin
    if (wc_write[0] && wc_full_n[0]) begin
      w_busy <= 1;
      w_base <= wc_din[0][31:0] >> 3;
      w_left <= wc_din[0][63:32];
      w_beat <= 0;
      w_reqs <= w_reqs + 1;
    end
    if (w_take) begin
      mem[w_base + w_beat] <= wd_din[0];
      w_beats <= w_beats + 1;
      if (w_last) w_busy <= 0;
      else w_beat <= w_beat + 1;
    end
    acks <= acks + (w_last ? 1 : 0) - ((wa_read[0] && wa_empty_n[0]) ? 1 : 0);
  end

  wire [63:0] dn_din [0:0];
  wire        dn_write [0:0], dn_full_n [0:0];
  assign dn_full_n[0] = 1'b1;

  spmw_top dut (
    .ap_clk(clk), .ap_rst_n(rst_n),
{ports}
  );

  integer cycle = 0, t0 = -1, t1 = -1, errs = 0, i, first_bad = -1;
  // what the head puts on the array's edge: a row a cycle unless it waits
  integer h_rows = 0, h_first = -1, h_last = -1;
  always @(posedge clk) begin
    cycle <= cycle + 1;
    if (launch && la_read[0]) launch <= 0;
    if (t0 >= 0 && t1 < 0 && dn_write[0]) t1 = cycle;
    if (dut.etap_e_in_bind_write[0]) begin
      h_rows <= h_rows + 1;
      if (h_first < 0) h_first = cycle;
      h_last = cycle;
    end
  end

  initial begin
    $readmemh("mem.hex", mem);
    $readmemh("want.hex", want);
    repeat (20) @(posedge clk);
    rst_n = 1;
    repeat (4) @(posedge clk);
    @(negedge clk); launch = 1; t0 = cycle;
    while (t1 < 0 && cycle < {limit}) @(posedge clk);
    repeat (4) @(posedge clk);
    for (i = 0; i < WORDS; i = i + 1)
      if (mem[i] !== want[i]) begin
        if (first_bad < 0) first_bad = i;
        errs = errs + 1;
      end
    $display("SPMWMEM RESULT errs=%0d total=%0d rows=%0d first=%0d paused=%0d rd_reqs=%0d rd_beats=%0d wr_reqs=%0d wr_beats=%0d first_bad=%0d timeout=%0d",
             errs, t1 - t0, h_rows, h_first - t0, h_last - h_first + 1 - h_rows,
             r_reqs, r_beats, w_reqs, w_beats, first_bad, (t1 < 0) ? 1 : 0);
    $display("SPMWMEM %s", (errs == 0 && t1 >= 0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
"""


def emit(stim_path, build, out, S, stall=0, tag=""):
    if stim_path:
        stim, launch = program(stim_path, S)
        what = f"M={stim['M']} K={stim['K']} N={stim['N']} tiles={stim['tiles']}"
    else:
        launch = Launch(S, mixed_gemms(S))
        what = f"gemms={len(launch.gemms)}"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/mem.hex", "w", encoding="utf-8") as handle:
        handle.write(hex_words(launch.memory))
    with open(f"{out}/want.hex", "w", encoding="utf-8") as handle:
        handle.write(hex_words(launch.final))
    names = boundary(os.path.join(build, "sim", "spmw_top.sv"))
    words = len(launch.memory) // 8
    with open(f"{out}/tbm.sv", "w", encoding="utf-8") as handle:
        limit = (8 * launch.rows + 400000) * (4 if stall else 1)
        handle.write(testbench(names, words, limit, stall))
    print(
        f"{tag} SPMWMEM PLAN size={S} {what} stall={stall}% "
        f"floor={launch.rows} words={words} "
        f"rd_reqs={len(launch.rd_cmd)} rd_beats={len(launch.rd_data)} "
        f"wr_reqs={len(launch.wr_cmd)} wr_beats={len(launch.wr_data)}"
    )


def simulate(build, out, tag):
    """Compile the build's units against the bench and run it."""
    sim = os.path.join(build, "sim")
    script = f"""set -e
cd {out}
for f in {sim}/*.dat; do [ -e "$f" ] && ln -sf "$f" . ; done
ls {sim}/*.sv | grep -v '/tb.sv$' > sv.f
ls {sim}/*.v > v.f
xvlog -sv -f sv.f tbm.sv > xvlog.log 2>&1
xvlog -f v.f >> xvlog.log 2>&1
xelab tbm -s tbmsim -timescale 1ns/1ps -L xpm -L unisims_ver -L unimacro_ver -L secureip > xelab.log 2>&1
xsim tbmsim -runall > xsim.log 2>&1
rm -rf xsim.dir tbmsim.wdb
"""
    done = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )
    if done.returncode:
        print(f"SPMWMEM {tag} tool failure:\n{done.stdout}{done.stderr}")
        return 1
    log = open(f"{out}/xsim.log", encoding="utf-8").read()
    for line in log.splitlines():
        if line.startswith("SPMWMEM"):
            print(f"{tag} {line}")
    # A bare-register link that was written before it was read says so.
    lost = log.count("SPMW OVERWRITE")
    if lost:
        print(f"{tag} SPMWMEM FAIL {lost} bare-register overwrite(s)")
    return 0 if "SPMWMEM PASS" in log and not lost else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--stim", help="a stimulus file: one GEMM")
    parser.add_argument("--mixed", action="store_true", help="five GEMMs instead")
    parser.add_argument("--build", required=True, help="a --cosim array build")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--stall", type=int, default=0, help="percent of cycles")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if bool(args.stim) == args.mixed:
        parser.error("give a stimulus file or --mixed")
    tag = args.tag or f"s{args.size}"
    emit(args.stim, args.build, args.out, args.size, args.stall, tag)
    sys.exit(simulate(args.build, args.out, tag))


if __name__ == "__main__":
    main()
