`timescale 1ns/1ps

module seq_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] op_in_dout,
  input  wire op_in_empty_n,
  output wire op_in_read,
  output wire [15:0] u_out_din,
  input  wire u_out_full_n,
  output wire u_out_write
);
  seq_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(op_in_dout),
      .v0_empty_n(op_in_empty_n),
      .v0_read(op_in_read),
      .v1_din(u_out_din),
      .v1_full_n(u_out_full_n),
      .v1_write(u_out_write));
endmodule
