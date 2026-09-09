`timescale 1ns/1ps

module mac_r3 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [31:0] op_in_dout,
  input  wire op_in_empty_n,
  output wire op_in_read,
  input  wire [31:0] w_in_dout,
  input  wire w_in_empty_n,
  output wire w_in_read,
  input  wire [7:0] a_in_dout,
  input  wire a_in_empty_n,
  output wire a_in_read,
  input  wire [31:0] p_in_dout,
  input  wire p_in_empty_n,
  output wire p_in_read,
  output wire [31:0] p_out_din,
  input  wire p_out_full_n,
  output wire p_out_write
);
  mac_r3_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(op_in_dout),
      .v0_empty_n(op_in_empty_n),
      .v0_read(op_in_read),
      .v1_dout(w_in_dout),
      .v1_empty_n(w_in_empty_n),
      .v1_read(w_in_read),
      .v2_dout(a_in_dout),
      .v2_empty_n(a_in_empty_n),
      .v2_read(a_in_read),
      .v3_dout(p_in_dout),
      .v3_empty_n(p_in_empty_n),
      .v3_read(p_in_read),
      .v4_din(p_out_din),
      .v4_full_n(p_out_full_n),
      .v4_write(p_out_write));
endmodule
