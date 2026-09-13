`timescale 1ns/1ps

module mac_r6 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [31:0] w_in_dout,
  input  wire w_in_empty_n,
  output wire w_in_read,
  input  wire [7:0] a_in_dout,
  input  wire a_in_empty_n,
  output wire a_in_read,
  output wire [31:0] p_out_din,
  input  wire p_out_full_n,
  output wire p_out_write
);
  mac_r6_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(w_in_dout),
      .v0_empty_n(w_in_empty_n),
      .v0_read(w_in_read),
      .v1_dout(a_in_dout),
      .v1_empty_n(a_in_empty_n),
      .v1_read(a_in_read),
      .v2_din(p_out_din),
      .v2_full_n(p_out_full_n),
      .v2_write(p_out_write));
endmodule
