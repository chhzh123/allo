`timescale 1ns/1ps

module etap_r1 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  output wire [15:0] a_out_din,
  input  wire a_out_full_n,
  output wire a_out_write,
  input  wire [135:0] e_in_dout,
  input  wire e_in_empty_n,
  output wire e_in_read,
  output wire [7:0] w_out_din,
  input  wire w_out_full_n,
  output wire w_out_write
);
  etap_r1_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_din(a_out_din),
      .v0_full_n(a_out_full_n),
      .v0_write(a_out_write),
      .v1_dout(e_in_dout),
      .v1_empty_n(e_in_empty_n),
      .v1_read(e_in_read),
      .v2_din(w_out_din),
      .v2_full_n(w_out_full_n),
      .v2_write(w_out_write));
endmodule
