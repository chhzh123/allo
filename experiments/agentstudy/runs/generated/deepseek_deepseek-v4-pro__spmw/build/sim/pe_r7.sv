`timescale 1ns/1ps

module pe_r7 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [7:0] b_in_dout,
  input  wire b_in_empty_n,
  output wire b_in_read,
  input  wire [7:0] a_in_dout,
  input  wire a_in_empty_n,
  output wire a_in_read,
  output wire [7:0] a_out_din,
  input  wire a_out_full_n,
  output wire a_out_write,
  output wire [31:0] c_out_din,
  input  wire c_out_full_n,
  output wire c_out_write
);
  pe_r7_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(b_in_dout),
      .v0_empty_n(b_in_empty_n),
      .v0_read(b_in_read),
      .v1_dout(a_in_dout),
      .v1_empty_n(a_in_empty_n),
      .v1_read(a_in_read),
      .v2_din(a_out_din),
      .v2_full_n(a_out_full_n),
      .v2_write(a_out_write),
      .v3_din(c_out_din),
      .v3_full_n(c_out_full_n),
      .v3_write(c_out_write));
endmodule
