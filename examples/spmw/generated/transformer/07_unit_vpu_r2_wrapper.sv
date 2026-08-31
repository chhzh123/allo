`timescale 1ns/1ps

module vpu_r2 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] b_dout,
  input  wire b_empty_n,
  output wire b_read,
  input  wire [31:0] op_in_dout,
  input  wire op_in_empty_n,
  output wire op_in_read,
  output wire [31:0] op_out_din,
  input  wire op_out_full_n,
  output wire op_out_write,
  input  wire [31:0] z_in_dout,
  input  wire z_in_empty_n,
  output wire z_in_read,
  output wire [31:0] y_out_din,
  input  wire y_out_full_n,
  output wire y_out_write
);
  vpu_r2_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(b_dout),
      .v0_empty_n(b_empty_n),
      .v0_read(b_read),
      .v1_dout(op_in_dout),
      .v1_empty_n(op_in_empty_n),
      .v1_read(op_in_read),
      .v2_din(op_out_din),
      .v2_full_n(op_out_full_n),
      .v2_write(op_out_write),
      .v3_dout(z_in_dout),
      .v3_empty_n(z_in_empty_n),
      .v3_read(z_in_read),
      .v4_din(y_out_din),
      .v4_full_n(y_out_full_n),
      .v4_write(y_out_write));
endmodule
