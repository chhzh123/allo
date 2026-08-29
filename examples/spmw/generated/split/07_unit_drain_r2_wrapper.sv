`timescale 1ns/1ps

module drain_r2 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [31:0] _pid0_dout,
  input  wire _pid0_empty_n,
  output wire _pid0_read,
  input  wire [31:0] _pid1_dout,
  input  wire _pid1_empty_n,
  output wire _pid1_read,
  input  wire [31:0] mine_dout,
  input  wire mine_empty_n,
  output wire mine_read,
  output wire [31:0] down_din,
  input  wire down_full_n,
  output wire down_write,
  input  wire [31:0] up_dout,
  input  wire up_empty_n,
  output wire up_read
);
  drain_r2_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(_pid0_dout),
      .v0_empty_n(_pid0_empty_n),
      .v0_read(_pid0_read),
      .v1_dout(_pid1_dout),
      .v1_empty_n(_pid1_empty_n),
      .v1_read(_pid1_read),
      .v2_dout(mine_dout),
      .v2_empty_n(mine_empty_n),
      .v2_read(mine_read),
      .v3_din(down_din),
      .v3_full_n(down_full_n),
      .v3_write(down_write),
      .v4_dout(up_dout),
      .v4_empty_n(up_empty_n),
      .v4_read(up_read));
endmodule
