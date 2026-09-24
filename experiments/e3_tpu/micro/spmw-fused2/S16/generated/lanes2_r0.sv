`timescale 1ns/1ps

module lanes2_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] b_dout,
  input  wire b_empty_n,
  output wire b_read,
  output wire [31:0] y0_out_din,
  input  wire y0_out_full_n,
  output wire y0_out_write,
  output wire [31:0] y1_out_din,
  input  wire y1_out_full_n,
  output wire y1_out_write,
  input  wire [31:0] z0_in_dout,
  input  wire z0_in_empty_n,
  output wire z0_in_read,
  input  wire [31:0] z1_in_dout,
  input  wire z1_in_empty_n,
  output wire z1_in_read
);
  lanes2_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(b_dout),
      .v0_empty_n(b_empty_n),
      .v0_read(b_read),
      .v1_din(y0_out_din),
      .v1_full_n(y0_out_full_n),
      .v1_write(y0_out_write),
      .v2_din(y1_out_din),
      .v2_full_n(y1_out_full_n),
      .v2_write(y1_out_write),
      .v3_dout(z0_in_dout),
      .v3_empty_n(z0_in_empty_n),
      .v3_read(z0_in_read),
      .v4_dout(z1_in_dout),
      .v4_empty_n(z1_in_empty_n),
      .v4_read(z1_in_read));
endmodule
