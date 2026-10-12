`timescale 1ns/1ps

module frame_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  output wire [15:0] out_din,
  input  wire out_full_n,
  output wire out_write,
  input  wire [7:0] raw_dout,
  input  wire raw_empty_n,
  output wire raw_read
);
  frame_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_din(out_din),
      .v0_full_n(out_full_n),
      .v0_write(out_write),
      .v1_dout(raw_dout),
      .v1_empty_n(raw_empty_n),
      .v1_read(raw_read));
endmodule
