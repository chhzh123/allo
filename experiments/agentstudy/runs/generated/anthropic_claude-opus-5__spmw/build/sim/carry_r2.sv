`timescale 1ns/1ps

module carry_r2 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [31:0] acc_in_dout,
  input  wire acc_in_empty_n,
  output wire acc_in_read,
  output wire [31:0] c_out_din,
  input  wire c_out_full_n,
  output wire c_out_write
);
  carry_r2_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(acc_in_dout),
      .v0_empty_n(acc_in_empty_n),
      .v0_read(acc_in_read),
      .v1_din(c_out_din),
      .v1_full_n(c_out_full_n),
      .v1_write(c_out_write));
endmodule
