`timescale 1ns/1ps

module spmw_const #(parameter DW = 32, parameter [63:0] VAL = 0) (
  output wire [DW-1:0] dout,
  output wire          empty_n,
  input  wire          read
);
  assign dout    = VAL[DW-1:0];
  assign empty_n = 1'b1;
endmodule
