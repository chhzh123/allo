`timescale 1ns/1ps

module pe_r3 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [7:0] a_in_dout,
  input  wire a_in_empty_n,
  output wire a_in_read,
  input  wire [7:0] b_in_dout,
  input  wire b_in_empty_n,
  output wire b_in_read,
  output wire [7:0] b_out_din,
  input  wire b_out_full_n,
  output wire b_out_write,
  input  wire [31:0] c_in_dout,
  input  wire c_in_empty_n,
  output wire c_in_read,
  output wire [31:0] c_out_din,
  input  wire c_out_full_n,
  output wire c_out_write
);
  pe_r3_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(a_in_dout),
      .v0_empty_n(a_in_empty_n),
      .v0_read(a_in_read),
      .v1_dout(b_in_dout),
      .v1_empty_n(b_in_empty_n),
      .v1_read(b_in_read),
      .v2_din(b_out_din),
      .v2_full_n(b_out_full_n),
      .v2_write(b_out_write),
      .v3_dout(c_in_dout),
      .v3_empty_n(c_in_empty_n),
      .v3_read(c_in_read),
      .v4_din(c_out_din),
      .v4_full_n(c_out_full_n),
      .v4_write(c_out_write));
endmodule
