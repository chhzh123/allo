`timescale 1ns/1ps

module pe_r2 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [7:0] a_in_dout,
  input  wire a_in_empty_n,
  output wire a_in_read,
  output wire [7:0] a_out_din,
  input  wire a_out_full_n,
  output wire a_out_write,
  output wire [31:0] p_out_din,
  input  wire p_out_full_n,
  output wire p_out_write,
  input  wire [7:0] w_in_dout,
  input  wire w_in_empty_n,
  output wire w_in_read,
  output wire [7:0] w_out_din,
  input  wire w_out_full_n,
  output wire w_out_write
);
  pe_r2_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(a_in_dout),
      .v0_empty_n(a_in_empty_n),
      .v0_read(a_in_read),
      .v1_din(a_out_din),
      .v1_full_n(a_out_full_n),
      .v1_write(a_out_write),
      .v2_din(p_out_din),
      .v2_full_n(p_out_full_n),
      .v2_write(p_out_write),
      .v3_dout(w_in_dout),
      .v3_empty_n(w_in_empty_n),
      .v3_read(w_in_read),
      .v4_din(w_out_din),
      .v4_full_n(w_out_full_n),
      .v4_write(w_out_write));
endmodule
