`timescale 1ns/1ps

module cross_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [31:0] _pid0_dout,
  input  wire _pid0_empty_n,
  output wire _pid0_read,
  input  wire [31:0] _pid1_dout,
  input  wire _pid1_empty_n,
  output wire _pid1_read,
  input  wire [63:0] a_in_dout,
  input  wire a_in_empty_n,
  output wire a_in_read,
  input  wire [63:0] b_in_dout,
  input  wire b_in_empty_n,
  output wire b_in_read,
  output wire [63:0] a_out_din,
  input  wire a_out_full_n,
  output wire a_out_write
);
  cross_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(_pid0_dout),
      .v0_empty_n(_pid0_empty_n),
      .v0_read(_pid0_read),
      .v1_dout(_pid1_dout),
      .v1_empty_n(_pid1_empty_n),
      .v1_read(_pid1_read),
      .v2_dout(a_in_dout),
      .v2_empty_n(a_in_empty_n),
      .v2_read(a_in_read),
      .v3_dout(b_in_dout),
      .v3_empty_n(b_in_empty_n),
      .v3_read(b_in_read),
      .v4_din(a_out_din),
      .v4_full_n(a_out_full_n),
      .v4_write(a_out_write));
endmodule
