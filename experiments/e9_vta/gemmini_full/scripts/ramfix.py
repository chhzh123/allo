# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ramfix.py <rtl dir> <out dir>: wide byte-masked memories as 64-bit columns.

Vivado infers a RAM for Gemmini's accumulator bank at 4x4, 2,048 x 128 bits
with a write mask per byte. At 8x8 and 16x16 the bank is 1,024 x 256 and
512 x 512 bits, 32 and 64 mask bits, and Vivado builds it from registers:
525,000 of them. In an ASIC flow this memory is a macro. Here it is rewritten
as columns of 64 bits and eight mask bits, the shape of a block RAM's
byte-write port, and nothing else in the RTL is touched. Prints the files it
rewrote, which the route reads in place of the originals.
"""
import os
import re
import sys

rtl, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for name in sorted(os.listdir(rtl)):
    hit = re.fullmatch(r"(mem_(\d+)x(\d+))\.sv", name)
    if not hit:
        continue
    text = open(os.path.join(rtl, name), encoding="utf-8").read()
    mask = re.search(r"input\s+\[(\d+):0\]\s+W0_mask", text)
    addr = re.search(r"input\s+\[(\d+):0\]\s+R0_addr", text)
    module, depth, width = hit.group(1), int(hit.group(2)), int(hit.group(3))
    if not mask or not addr or width <= 128:
        continue
    masks, aw = int(mask.group(1)) + 1, int(addr.group(1)) + 1
    assert width == 8 * masks and width % 64 == 0, name
    cols = width // 64
    body = f"""// Rewritten by ramfix.py from firtool's {name}: the same memory as
// {cols} columns of 64 bits, so that Vivado maps it to block RAM.
module {module}_col(
  input  [{aw - 1}:0] R0_addr,
  input         R0_clk,
  output [63:0] R0_data,
  input  [{aw - 1}:0] W0_addr,
  input         W0_en,
  input         W0_clk,
  input  [63:0] W0_data,
  input  [7:0]  W0_mask
);
  (* ram_style = "block" *) reg [63:0] Memory[0:{depth - 1}];
  reg [{aw - 1}:0] raddr;
  integer b;
  always @(posedge R0_clk) raddr <= R0_addr;
  always @(posedge W0_clk)
    for (b = 0; b < 8; b = b + 1)
      if (W0_en & W0_mask[b]) Memory[W0_addr][8*b +: 8] <= W0_data[8*b +: 8];
  assign R0_data = Memory[raddr];
endmodule

module {module}(
  input  [{aw - 1}:0] R0_addr,
  input         R0_en,
  input         R0_clk,
  output [{width - 1}:0] R0_data,
  input  [{aw - 1}:0] W0_addr,
  input         W0_en,
  input         W0_clk,
  input  [{width - 1}:0] W0_data,
  input  [{masks - 1}:0] W0_mask
);
  genvar g;
  generate for (g = 0; g < {cols}; g = g + 1) begin : col
    {module}_col m (
      .R0_addr(R0_addr), .R0_clk(R0_clk), .R0_data(R0_data[64*g +: 64]),
      .W0_addr(W0_addr), .W0_en(W0_en), .W0_clk(W0_clk),
      .W0_data(W0_data[64*g +: 64]), .W0_mask(W0_mask[8*g +: 8]));
  end endgenerate
endmodule
"""
    with open(os.path.join(out, name), "w", encoding="utf-8") as handle:
        handle.write(body)
    print(name)
