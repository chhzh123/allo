`timescale 1ns/1ps

module feed_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [31:0] _pid0_dout,
  input  wire _pid0_empty_n,
  output wire _pid0_read,
  input  wire [63:0] up_dout,
  input  wire up_empty_n,
  output wire up_read,
  output wire [7:0] lane_din,
  input  wire lane_full_n,
  output wire lane_write,
  output wire [63:0] down_din,
  input  wire down_full_n,
  output wire down_write
);
  feed_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(_pid0_dout),
      .v0_empty_n(_pid0_empty_n),
      .v0_read(_pid0_read),
      .v1_dout(up_dout),
      .v1_empty_n(up_empty_n),
      .v1_read(up_read),
      .v2_din(lane_din),
      .v2_full_n(lane_full_n),
      .v2_write(lane_write),
      .v3_din(down_din),
      .v3_full_n(down_full_n),
      .v3_write(down_write));
endmodule
