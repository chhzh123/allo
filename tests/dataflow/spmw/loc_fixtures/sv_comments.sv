// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// Fixture for scripts/spmw_loc2.py: SystemVerilog comments and literals.
/* A block comment
   does not count. */
module keep_counter #(parameter WIDTH = 8) ( // KEEP
  input  logic clk, // KEEP
  output logic [WIDTH-1:0] q // KEEP
); // KEEP
  logic [3:0] nibble = 4'b1010; // KEEP: a lone apostrophe is a base marker, not a delimiter
  string msg = "// not a comment /* nor this */"; // KEEP
  /* inline */ always_ff @(posedge clk) q <= q + 1'b1; // KEEP
  initial $display("it's fine: %s", msg); // KEEP: an apostrophe inside a string
endmodule // KEEP
