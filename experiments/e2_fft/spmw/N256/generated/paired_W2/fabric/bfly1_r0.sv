`timescale 1ns/1ps

module bfly1_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] a_in_dout,
  input  wire a_in_empty_n,
  output wire a_in_read,
  input  wire [63:0] b_in_dout,
  input  wire b_in_empty_n,
  output wire b_in_read,
  output wire [63:0] a_out_din,
  input  wire a_out_full_n,
  output wire a_out_write,
  output wire [63:0] b_out_din,
  input  wire b_out_full_n,
  output wire b_out_write
);
  bfly1_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(a_in_dout),
      .v0_empty_n(a_in_empty_n),
      .v0_read(a_in_read),
      .v1_dout(b_in_dout),
      .v1_empty_n(b_in_empty_n),
      .v1_read(b_in_read),
      .v2_din(a_out_din),
      .v2_full_n(a_out_full_n),
      .v2_write(a_out_write),
      .v3_din(b_out_din),
      .v3_full_n(b_out_full_n),
      .v3_write(b_out_write));
endmodule
