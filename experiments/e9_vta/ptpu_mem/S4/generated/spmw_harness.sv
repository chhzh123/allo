`timescale 1ns/1ps

module spmw_harness (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire        start,
  output reg  [31:0] sig
);
  // A maximal-length LFSR per *channel*, not one shared.
  //
  // Sharing one was the first thing measured and it was wrong: a single
  // net driving every channel of every family is a fanout of hundreds
  // across the whole die, and it became the critical path -- 3.877 ns of
  // route against 0.080 ns of logic, sourced at `lfsr_reg` and sinking
  // in a MAC cell. That measured the harness, not the fabric. One small
  // LFSR per channel is a few flip-flops each and keeps every source
  // beside its load.

  wire [63:0] req4_launch_bind_dout [0:0];
  wire req4_launch_bind_empty_n [0:0];
  wire req4_launch_bind_read [0:0];
  genvar g_req4_launch_bind;
  generate
    for (g_req4_launch_bind = 0; g_req4_launch_bind < 1; g_req4_launch_bind = g_req4_launch_bind + 1) begin : gen_req4_launch_bind
      reg [31:0] lf_req4_launch_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_req4_launch_bind <= 32'h1 + g_req4_launch_bind;
        else if (start) lf_req4_launch_bind <= {lf_req4_launch_bind[30:0], lf_req4_launch_bind[31]^lf_req4_launch_bind[21]^lf_req4_launch_bind[1]^lf_req4_launch_bind[0]};
      assign req4_launch_bind_dout[g_req4_launch_bind] = {2{lf_req4_launch_bind}};
      assign req4_launch_bind_empty_n[g_req4_launch_bind] = start;
    end
  endgenerate
  wire [63:0] req4_rd_cmd_bind_din [0:0];
  wire req4_rd_cmd_bind_write [0:0];
  wire req4_rd_cmd_bind_full_n [0:0];
  genvar g_req4_rd_cmd_bind;
  generate
    for (g_req4_rd_cmd_bind = 0; g_req4_rd_cmd_bind < 1; g_req4_rd_cmd_bind = g_req4_rd_cmd_bind + 1) begin : gen_req4_rd_cmd_bind
      assign req4_rd_cmd_bind_full_n[g_req4_rd_cmd_bind] = 1'b1;
    end
  endgenerate
  wire [63:0] deal4_rd_data_bind_dout [0:0];
  wire deal4_rd_data_bind_empty_n [0:0];
  wire deal4_rd_data_bind_read [0:0];
  genvar g_deal4_rd_data_bind;
  generate
    for (g_deal4_rd_data_bind = 0; g_deal4_rd_data_bind < 1; g_deal4_rd_data_bind = g_deal4_rd_data_bind + 1) begin : gen_deal4_rd_data_bind
      reg [31:0] lf_deal4_rd_data_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_deal4_rd_data_bind <= 32'h1 + g_deal4_rd_data_bind;
        else if (start) lf_deal4_rd_data_bind <= {lf_deal4_rd_data_bind[30:0], lf_deal4_rd_data_bind[31]^lf_deal4_rd_data_bind[21]^lf_deal4_rd_data_bind[1]^lf_deal4_rd_data_bind[0]};
      assign deal4_rd_data_bind_dout[g_deal4_rd_data_bind] = {2{lf_deal4_rd_data_bind}};
      assign deal4_rd_data_bind_empty_n[g_deal4_rd_data_bind] = start;
    end
  endgenerate
  wire [63:0] wreq4_wr_ack_bind_dout [0:0];
  wire wreq4_wr_ack_bind_empty_n [0:0];
  wire wreq4_wr_ack_bind_read [0:0];
  genvar g_wreq4_wr_ack_bind;
  generate
    for (g_wreq4_wr_ack_bind = 0; g_wreq4_wr_ack_bind < 1; g_wreq4_wr_ack_bind = g_wreq4_wr_ack_bind + 1) begin : gen_wreq4_wr_ack_bind
      reg [31:0] lf_wreq4_wr_ack_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_wreq4_wr_ack_bind <= 32'h1 + g_wreq4_wr_ack_bind;
        else if (start) lf_wreq4_wr_ack_bind <= {lf_wreq4_wr_ack_bind[30:0], lf_wreq4_wr_ack_bind[31]^lf_wreq4_wr_ack_bind[21]^lf_wreq4_wr_ack_bind[1]^lf_wreq4_wr_ack_bind[0]};
      assign wreq4_wr_ack_bind_dout[g_wreq4_wr_ack_bind] = {2{lf_wreq4_wr_ack_bind}};
      assign wreq4_wr_ack_bind_empty_n[g_wreq4_wr_ack_bind] = start;
    end
  endgenerate
  wire [63:0] wreq4_wr_cmd_bind_din [0:0];
  wire wreq4_wr_cmd_bind_write [0:0];
  wire wreq4_wr_cmd_bind_full_n [0:0];
  genvar g_wreq4_wr_cmd_bind;
  generate
    for (g_wreq4_wr_cmd_bind = 0; g_wreq4_wr_cmd_bind < 1; g_wreq4_wr_cmd_bind = g_wreq4_wr_cmd_bind + 1) begin : gen_wreq4_wr_cmd_bind
      assign wreq4_wr_cmd_bind_full_n[g_wreq4_wr_cmd_bind] = 1'b1;
    end
  endgenerate
  wire [63:0] wreq4_done_bind_din [0:0];
  wire wreq4_done_bind_write [0:0];
  wire wreq4_done_bind_full_n [0:0];
  genvar g_wreq4_done_bind;
  generate
    for (g_wreq4_done_bind = 0; g_wreq4_done_bind < 1; g_wreq4_done_bind = g_wreq4_done_bind + 1) begin : gen_wreq4_done_bind
      assign wreq4_done_bind_full_n[g_wreq4_done_bind] = 1'b1;
    end
  endgenerate
  wire [63:0] pack4_wr_data_bind_din [0:0];
  wire pack4_wr_data_bind_write [0:0];
  wire pack4_wr_data_bind_full_n [0:0];
  genvar g_pack4_wr_data_bind;
  generate
    for (g_pack4_wr_data_bind = 0; g_pack4_wr_data_bind < 1; g_pack4_wr_data_bind = g_pack4_wr_data_bind + 1) begin : gen_pack4_wr_data_bind
      assign pack4_wr_data_bind_full_n[g_pack4_wr_data_bind] = 1'b1;
    end
  endgenerate

  spmw_top dut (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .req4_launch_bind_dout(req4_launch_bind_dout),
      .req4_launch_bind_empty_n(req4_launch_bind_empty_n),
      .req4_launch_bind_read(req4_launch_bind_read),
      .req4_rd_cmd_bind_din(req4_rd_cmd_bind_din),
      .req4_rd_cmd_bind_write(req4_rd_cmd_bind_write),
      .req4_rd_cmd_bind_full_n(req4_rd_cmd_bind_full_n),
      .deal4_rd_data_bind_dout(deal4_rd_data_bind_dout),
      .deal4_rd_data_bind_empty_n(deal4_rd_data_bind_empty_n),
      .deal4_rd_data_bind_read(deal4_rd_data_bind_read),
      .wreq4_wr_ack_bind_dout(wreq4_wr_ack_bind_dout),
      .wreq4_wr_ack_bind_empty_n(wreq4_wr_ack_bind_empty_n),
      .wreq4_wr_ack_bind_read(wreq4_wr_ack_bind_read),
      .wreq4_wr_cmd_bind_din(wreq4_wr_cmd_bind_din),
      .wreq4_wr_cmd_bind_write(wreq4_wr_cmd_bind_write),
      .wreq4_wr_cmd_bind_full_n(wreq4_wr_cmd_bind_full_n),
      .wreq4_done_bind_din(wreq4_done_bind_din),
      .wreq4_done_bind_write(wreq4_done_bind_write),
      .wreq4_done_bind_full_n(wreq4_done_bind_full_n),
      .pack4_wr_data_bind_din(pack4_wr_data_bind_din),
      .pack4_wr_data_bind_write(pack4_wr_data_bind_write),
      .pack4_wr_data_bind_full_n(pack4_wr_data_bind_full_n));

  // Fold every output into one register: a dangling result is a result
  // synthesis is entitled to delete.
  always @(posedge ap_clk)
    if (!ap_rst_n) sig <= 32'b0;
    else sig <= sig
      ^ {31'b0, req4_launch_bind_read[0]}
      ^ req4_rd_cmd_bind_din[0][31:0]
      ^ {31'b0, req4_rd_cmd_bind_write[0]}
      ^ {31'b0, deal4_rd_data_bind_read[0]}
      ^ {31'b0, wreq4_wr_ack_bind_read[0]}
      ^ wreq4_wr_cmd_bind_din[0][31:0]
      ^ {31'b0, wreq4_wr_cmd_bind_write[0]}
      ^ wreq4_done_bind_din[0][31:0]
      ^ {31'b0, wreq4_done_bind_write[0]}
      ^ pack4_wr_data_bind_din[0][31:0]
      ^ {31'b0, pack4_wr_data_bind_write[0]}
      ;
endmodule
