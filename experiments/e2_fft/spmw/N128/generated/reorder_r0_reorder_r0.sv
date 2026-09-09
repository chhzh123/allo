`timescale 1ns/1ps

module reorder_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] x_in_dout,
  input  wire x_in_empty_n,
  output wire x_in_read,
  output wire [63:0] y_out_din,
  input  wire y_out_full_n,
  output wire y_out_write
);
  reorder_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(x_in_dout),
      .v0_empty_n(x_in_empty_n),
      .v0_read(x_in_read),
      .v1_din(y_out_din),
      .v1_full_n(y_out_full_n),
      .v1_write(y_out_write));
endmodule
