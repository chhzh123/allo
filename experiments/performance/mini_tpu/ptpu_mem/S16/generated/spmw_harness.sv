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

  wire [63:0] req16_launch_bind_dout [0:0];
  wire req16_launch_bind_empty_n [0:0];
  wire req16_launch_bind_read [0:0];
  genvar g_req16_launch_bind;
  generate
    for (g_req16_launch_bind = 0; g_req16_launch_bind < 1; g_req16_launch_bind = g_req16_launch_bind + 1) begin : gen_req16_launch_bind
      reg [31:0] lf_req16_launch_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_req16_launch_bind <= 32'h1 + g_req16_launch_bind;
        else if (start) lf_req16_launch_bind <= {lf_req16_launch_bind[30:0], lf_req16_launch_bind[31]^lf_req16_launch_bind[21]^lf_req16_launch_bind[1]^lf_req16_launch_bind[0]};
      assign req16_launch_bind_dout[g_req16_launch_bind] = {2{lf_req16_launch_bind}};
      assign req16_launch_bind_empty_n[g_req16_launch_bind] = start;
    end
  endgenerate
  wire [63:0] req16_rd_cmd_bind_din [0:0];
  wire req16_rd_cmd_bind_write [0:0];
  wire req16_rd_cmd_bind_full_n [0:0];
  genvar g_req16_rd_cmd_bind;
  generate
    for (g_req16_rd_cmd_bind = 0; g_req16_rd_cmd_bind < 1; g_req16_rd_cmd_bind = g_req16_rd_cmd_bind + 1) begin : gen_req16_rd_cmd_bind
      assign req16_rd_cmd_bind_full_n[g_req16_rd_cmd_bind] = 1'b1;
    end
  endgenerate
  wire [63:0] deal16_rd_data_bind_dout [0:0];
  wire deal16_rd_data_bind_empty_n [0:0];
  wire deal16_rd_data_bind_read [0:0];
  genvar g_deal16_rd_data_bind;
  generate
    for (g_deal16_rd_data_bind = 0; g_deal16_rd_data_bind < 1; g_deal16_rd_data_bind = g_deal16_rd_data_bind + 1) begin : gen_deal16_rd_data_bind
      reg [31:0] lf_deal16_rd_data_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_deal16_rd_data_bind <= 32'h1 + g_deal16_rd_data_bind;
        else if (start) lf_deal16_rd_data_bind <= {lf_deal16_rd_data_bind[30:0], lf_deal16_rd_data_bind[31]^lf_deal16_rd_data_bind[21]^lf_deal16_rd_data_bind[1]^lf_deal16_rd_data_bind[0]};
      assign deal16_rd_data_bind_dout[g_deal16_rd_data_bind] = {2{lf_deal16_rd_data_bind}};
      assign deal16_rd_data_bind_empty_n[g_deal16_rd_data_bind] = start;
    end
  endgenerate
  wire [63:0] wreq16_wr_ack_bind_dout [0:0];
  wire wreq16_wr_ack_bind_empty_n [0:0];
  wire wreq16_wr_ack_bind_read [0:0];
  genvar g_wreq16_wr_ack_bind;
  generate
    for (g_wreq16_wr_ack_bind = 0; g_wreq16_wr_ack_bind < 1; g_wreq16_wr_ack_bind = g_wreq16_wr_ack_bind + 1) begin : gen_wreq16_wr_ack_bind
      reg [31:0] lf_wreq16_wr_ack_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_wreq16_wr_ack_bind <= 32'h1 + g_wreq16_wr_ack_bind;
        else if (start) lf_wreq16_wr_ack_bind <= {lf_wreq16_wr_ack_bind[30:0], lf_wreq16_wr_ack_bind[31]^lf_wreq16_wr_ack_bind[21]^lf_wreq16_wr_ack_bind[1]^lf_wreq16_wr_ack_bind[0]};
      assign wreq16_wr_ack_bind_dout[g_wreq16_wr_ack_bind] = {2{lf_wreq16_wr_ack_bind}};
      assign wreq16_wr_ack_bind_empty_n[g_wreq16_wr_ack_bind] = start;
    end
  endgenerate
  wire [63:0] wreq16_wr_cmd_bind_din [0:0];
  wire wreq16_wr_cmd_bind_write [0:0];
  wire wreq16_wr_cmd_bind_full_n [0:0];
  genvar g_wreq16_wr_cmd_bind;
  generate
    for (g_wreq16_wr_cmd_bind = 0; g_wreq16_wr_cmd_bind < 1; g_wreq16_wr_cmd_bind = g_wreq16_wr_cmd_bind + 1) begin : gen_wreq16_wr_cmd_bind
      assign wreq16_wr_cmd_bind_full_n[g_wreq16_wr_cmd_bind] = 1'b1;
    end
  endgenerate
  wire [63:0] wreq16_done_bind_din [0:0];
  wire wreq16_done_bind_write [0:0];
  wire wreq16_done_bind_full_n [0:0];
  genvar g_wreq16_done_bind;
  generate
    for (g_wreq16_done_bind = 0; g_wreq16_done_bind < 1; g_wreq16_done_bind = g_wreq16_done_bind + 1) begin : gen_wreq16_done_bind
      assign wreq16_done_bind_full_n[g_wreq16_done_bind] = 1'b1;
    end
  endgenerate
  wire [63:0] pack16_wr_data_bind_din [0:0];
  wire pack16_wr_data_bind_write [0:0];
  wire pack16_wr_data_bind_full_n [0:0];
  genvar g_pack16_wr_data_bind;
  generate
    for (g_pack16_wr_data_bind = 0; g_pack16_wr_data_bind < 1; g_pack16_wr_data_bind = g_pack16_wr_data_bind + 1) begin : gen_pack16_wr_data_bind
      assign pack16_wr_data_bind_full_n[g_pack16_wr_data_bind] = 1'b1;
    end
  endgenerate

  spmw_top dut (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .req16_launch_bind_dout(req16_launch_bind_dout),
      .req16_launch_bind_empty_n(req16_launch_bind_empty_n),
      .req16_launch_bind_read(req16_launch_bind_read),
      .req16_rd_cmd_bind_din(req16_rd_cmd_bind_din),
      .req16_rd_cmd_bind_write(req16_rd_cmd_bind_write),
      .req16_rd_cmd_bind_full_n(req16_rd_cmd_bind_full_n),
      .deal16_rd_data_bind_dout(deal16_rd_data_bind_dout),
      .deal16_rd_data_bind_empty_n(deal16_rd_data_bind_empty_n),
      .deal16_rd_data_bind_read(deal16_rd_data_bind_read),
      .wreq16_wr_ack_bind_dout(wreq16_wr_ack_bind_dout),
      .wreq16_wr_ack_bind_empty_n(wreq16_wr_ack_bind_empty_n),
      .wreq16_wr_ack_bind_read(wreq16_wr_ack_bind_read),
      .wreq16_wr_cmd_bind_din(wreq16_wr_cmd_bind_din),
      .wreq16_wr_cmd_bind_write(wreq16_wr_cmd_bind_write),
      .wreq16_wr_cmd_bind_full_n(wreq16_wr_cmd_bind_full_n),
      .wreq16_done_bind_din(wreq16_done_bind_din),
      .wreq16_done_bind_write(wreq16_done_bind_write),
      .wreq16_done_bind_full_n(wreq16_done_bind_full_n),
      .pack16_wr_data_bind_din(pack16_wr_data_bind_din),
      .pack16_wr_data_bind_write(pack16_wr_data_bind_write),
      .pack16_wr_data_bind_full_n(pack16_wr_data_bind_full_n));

  // Fold every output into one register: a dangling result is a result
  // synthesis is entitled to delete.
  always @(posedge ap_clk)
    if (!ap_rst_n) sig <= 32'b0;
    else sig <= sig
      ^ {31'b0, req16_launch_bind_read[0]}
      ^ req16_rd_cmd_bind_din[0][31:0]
      ^ {31'b0, req16_rd_cmd_bind_write[0]}
      ^ {31'b0, deal16_rd_data_bind_read[0]}
      ^ {31'b0, wreq16_wr_ack_bind_read[0]}
      ^ wreq16_wr_cmd_bind_din[0][31:0]
      ^ {31'b0, wreq16_wr_cmd_bind_write[0]}
      ^ wreq16_done_bind_din[0][31:0]
      ^ {31'b0, wreq16_done_bind_write[0]}
      ^ pack16_wr_data_bind_din[0][31:0]
      ^ {31'b0, pack16_wr_data_bind_write[0]}
      ;
endmodule
