`timescale 1ns/1ps

module feed_r2 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  output wire [7:0] lane_din,
  input  wire lane_full_n,
  output wire lane_write,
  input  wire [31:0] up_dout,
  input  wire up_empty_n,
  output wire up_read,
  input  wire [31:0] _pid0_dout,
  input  wire _pid0_empty_n,
  output wire _pid0_read
);
  feed_r2_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_din(lane_din),
      .v0_full_n(lane_full_n),
      .v0_write(lane_write),
      .v1_dout(up_dout),
      .v1_empty_n(up_empty_n),
      .v1_read(up_read),
      .v2_dout(_pid0_dout),
      .v2_empty_n(_pid0_empty_n),
      .v2_read(_pid0_read));
endmodule
