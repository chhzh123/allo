`timescale 1ns/1ps

module act_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [31:0] z_in_dout,
  input  wire z_in_empty_n,
  output wire z_in_read,
  output wire [7:0] y_out_din,
  input  wire y_out_full_n,
  output wire y_out_write
);
  act_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(z_in_dout),
      .v0_empty_n(z_in_empty_n),
      .v0_read(z_in_read),
      .v1_din(y_out_din),
      .v1_full_n(y_out_full_n),
      .v1_write(y_out_write));
endmodule
