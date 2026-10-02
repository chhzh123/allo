`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] req8_launch_bind_dout [0:0],
  input  wire req8_launch_bind_empty_n [0:0],
  output wire req8_launch_bind_read [0:0],
  output wire [63:0] req8_rd_cmd_bind_din [0:0],
  output wire req8_rd_cmd_bind_write [0:0],
  input  wire req8_rd_cmd_bind_full_n [0:0],
  input  wire [63:0] deal8_rd_data_bind_dout [0:0],
  input  wire deal8_rd_data_bind_empty_n [0:0],
  output wire deal8_rd_data_bind_read [0:0],
  input  wire [63:0] wreq8_wr_ack_bind_dout [0:0],
  input  wire wreq8_wr_ack_bind_empty_n [0:0],
  output wire wreq8_wr_ack_bind_read [0:0],
  output wire [63:0] wreq8_wr_cmd_bind_din [0:0],
  output wire wreq8_wr_cmd_bind_write [0:0],
  input  wire wreq8_wr_cmd_bind_full_n [0:0],
  output wire [63:0] wreq8_done_bind_din [0:0],
  output wire wreq8_done_bind_write [0:0],
  input  wire wreq8_done_bind_full_n [0:0],
  output wire [63:0] pack8_wr_data_bind_din [0:0],
  output wire pack8_wr_data_bind_write [0:0],
  input  wire pack8_wr_data_bind_full_n [0:0]
);
  // family pe_a_out_a_in: 64 channel(s), 16-bit, depth 0
  wire [15:0] pe_a_out_a_in_din [0:63];
  wire [15:0] pe_a_out_a_in_dout [0:63];
  wire pe_a_out_a_in_full_n [0:63];
  wire pe_a_out_a_in_write [0:63];
  wire pe_a_out_a_in_empty_n [0:63];
  wire pe_a_out_a_in_read [0:63];
  genvar pe_a_out_a_in_i;
  generate
    for (pe_a_out_a_in_i = 0; pe_a_out_a_in_i < 64; pe_a_out_a_in_i = pe_a_out_a_in_i + 1) begin : g_pe_a_out_a_in
      spmw_fifo #(.DW(16), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_a_out_a_in_din[pe_a_out_a_in_i]), .full_n(pe_a_out_a_in_full_n[pe_a_out_a_in_i]), .write(pe_a_out_a_in_write[pe_a_out_a_in_i]), .dout(pe_a_out_a_in_dout[pe_a_out_a_in_i]), .empty_n(pe_a_out_a_in_empty_n[pe_a_out_a_in_i]), .read(pe_a_out_a_in_read[pe_a_out_a_in_i]));
    end
  endgenerate
  // family pe_w_out_w_in: 64 channel(s), 8-bit, depth 0
  wire [7:0] pe_w_out_w_in_din [0:63];
  wire [7:0] pe_w_out_w_in_dout [0:63];
  wire pe_w_out_w_in_full_n [0:63];
  wire pe_w_out_w_in_write [0:63];
  wire pe_w_out_w_in_empty_n [0:63];
  wire pe_w_out_w_in_read [0:63];
  genvar pe_w_out_w_in_i;
  generate
    for (pe_w_out_w_in_i = 0; pe_w_out_w_in_i < 64; pe_w_out_w_in_i = pe_w_out_w_in_i + 1) begin : g_pe_w_out_w_in
      spmw_fifo #(.DW(8), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_w_out_w_in_din[pe_w_out_w_in_i]), .full_n(pe_w_out_w_in_full_n[pe_w_out_w_in_i]), .write(pe_w_out_w_in_write[pe_w_out_w_in_i]), .dout(pe_w_out_w_in_dout[pe_w_out_w_in_i]), .empty_n(pe_w_out_w_in_empty_n[pe_w_out_w_in_i]), .read(pe_w_out_w_in_read[pe_w_out_w_in_i]));
    end
  endgenerate
  // family pe_p_out_p_in: 64 channel(s), 32-bit, depth 0
  wire [31:0] pe_p_out_p_in_din [0:63];
  wire [31:0] pe_p_out_p_in_dout [0:63];
  wire pe_p_out_p_in_full_n [0:63];
  wire pe_p_out_p_in_write [0:63];
  wire pe_p_out_p_in_empty_n [0:63];
  wire pe_p_out_p_in_read [0:63];
  genvar pe_p_out_p_in_i;
  generate
    for (pe_p_out_p_in_i = 0; pe_p_out_p_in_i < 64; pe_p_out_p_in_i = pe_p_out_p_in_i + 1) begin : g_pe_p_out_p_in
      spmw_fifo #(.DW(32), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_p_out_p_in_din[pe_p_out_p_in_i]), .full_n(pe_p_out_p_in_full_n[pe_p_out_p_in_i]), .write(pe_p_out_p_in_write[pe_p_out_p_in_i]), .dout(pe_p_out_p_in_dout[pe_p_out_p_in_i]), .empty_n(pe_p_out_p_in_empty_n[pe_p_out_p_in_i]), .read(pe_p_out_p_in_read[pe_p_out_p_in_i]));
    end
  endgenerate
  // family pe_a_in_bind: 8 channel(s), 16-bit, depth 0
  wire [15:0] pe_a_in_bind_din [0:7];
  wire [15:0] pe_a_in_bind_dout [0:7];
  wire pe_a_in_bind_full_n [0:7];
  wire pe_a_in_bind_write [0:7];
  wire pe_a_in_bind_empty_n [0:7];
  wire pe_a_in_bind_read [0:7];
  genvar pe_a_in_bind_i;
  generate
    for (pe_a_in_bind_i = 0; pe_a_in_bind_i < 8; pe_a_in_bind_i = pe_a_in_bind_i + 1) begin : g_pe_a_in_bind
      spmw_fifo #(.DW(16), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_a_in_bind_din[pe_a_in_bind_i]), .full_n(pe_a_in_bind_full_n[pe_a_in_bind_i]), .write(pe_a_in_bind_write[pe_a_in_bind_i]), .dout(pe_a_in_bind_dout[pe_a_in_bind_i]), .empty_n(pe_a_in_bind_empty_n[pe_a_in_bind_i]), .read(pe_a_in_bind_read[pe_a_in_bind_i]));
    end
  endgenerate
  // family pe_w_in_bind: 8 channel(s), 8-bit, depth 0
  wire [7:0] pe_w_in_bind_din [0:7];
  wire [7:0] pe_w_in_bind_dout [0:7];
  wire pe_w_in_bind_full_n [0:7];
  wire pe_w_in_bind_write [0:7];
  wire pe_w_in_bind_empty_n [0:7];
  wire pe_w_in_bind_read [0:7];
  genvar pe_w_in_bind_i;
  generate
    for (pe_w_in_bind_i = 0; pe_w_in_bind_i < 8; pe_w_in_bind_i = pe_w_in_bind_i + 1) begin : g_pe_w_in_bind
      spmw_fifo #(.DW(8), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_w_in_bind_din[pe_w_in_bind_i]), .full_n(pe_w_in_bind_full_n[pe_w_in_bind_i]), .write(pe_w_in_bind_write[pe_w_in_bind_i]), .dout(pe_w_in_bind_dout[pe_w_in_bind_i]), .empty_n(pe_w_in_bind_empty_n[pe_w_in_bind_i]), .read(pe_w_in_bind_read[pe_w_in_bind_i]));
    end
  endgenerate
  // family lane_z_in_bind: 8 channel(s), 32-bit, depth 2
  wire [31:0] lane_z_in_bind_din [0:7];
  wire [31:0] lane_z_in_bind_dout [0:7];
  wire lane_z_in_bind_full_n [0:7];
  wire lane_z_in_bind_write [0:7];
  wire lane_z_in_bind_empty_n [0:7];
  wire lane_z_in_bind_read [0:7];
  genvar lane_z_in_bind_i;
  generate
    for (lane_z_in_bind_i = 0; lane_z_in_bind_i < 8; lane_z_in_bind_i = lane_z_in_bind_i + 1) begin : g_lane_z_in_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(lane_z_in_bind_din[lane_z_in_bind_i]), .full_n(lane_z_in_bind_full_n[lane_z_in_bind_i]), .write(lane_z_in_bind_write[lane_z_in_bind_i]), .dout(lane_z_in_bind_dout[lane_z_in_bind_i]), .empty_n(lane_z_in_bind_empty_n[lane_z_in_bind_i]), .read(lane_z_in_bind_read[lane_z_in_bind_i]));
    end
  endgenerate
  // family etap_e_out_e_in: 8 channel(s), 136-bit, depth 0
  wire [135:0] etap_e_out_e_in_din [0:7];
  wire [135:0] etap_e_out_e_in_dout [0:7];
  wire etap_e_out_e_in_full_n [0:7];
  wire etap_e_out_e_in_write [0:7];
  wire etap_e_out_e_in_empty_n [0:7];
  wire etap_e_out_e_in_read [0:7];
  genvar etap_e_out_e_in_i;
  generate
    for (etap_e_out_e_in_i = 0; etap_e_out_e_in_i < 8; etap_e_out_e_in_i = etap_e_out_e_in_i + 1) begin : g_etap_e_out_e_in
      spmw_fifo #(.DW(136), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(etap_e_out_e_in_din[etap_e_out_e_in_i]), .full_n(etap_e_out_e_in_full_n[etap_e_out_e_in_i]), .write(etap_e_out_e_in_write[etap_e_out_e_in_i]), .dout(etap_e_out_e_in_dout[etap_e_out_e_in_i]), .empty_n(etap_e_out_e_in_empty_n[etap_e_out_e_in_i]), .read(etap_e_out_e_in_read[etap_e_out_e_in_i]));
    end
  endgenerate
  // family etap_e_in_bind: 1 channel(s), 136-bit, depth 0
  wire [135:0] etap_e_in_bind_din [0:0];
  wire [135:0] etap_e_in_bind_dout [0:0];
  wire etap_e_in_bind_full_n [0:0];
  wire etap_e_in_bind_write [0:0];
  wire etap_e_in_bind_empty_n [0:0];
  wire etap_e_in_bind_read [0:0];
  genvar etap_e_in_bind_i;
  generate
    for (etap_e_in_bind_i = 0; etap_e_in_bind_i < 1; etap_e_in_bind_i = etap_e_in_bind_i + 1) begin : g_etap_e_in_bind
      spmw_fifo #(.DW(136), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(etap_e_in_bind_din[etap_e_in_bind_i]), .full_n(etap_e_in_bind_full_n[etap_e_in_bind_i]), .write(etap_e_in_bind_write[etap_e_in_bind_i]), .dout(etap_e_in_bind_dout[etap_e_in_bind_i]), .empty_n(etap_e_in_bind_empty_n[etap_e_in_bind_i]), .read(etap_e_in_bind_read[etap_e_in_bind_i]));
    end
  endgenerate
  // family deal8_tag_in_bind: 1 channel(s), 32-bit, depth 4
  wire [31:0] deal8_tag_in_bind_din [0:0];
  wire [31:0] deal8_tag_in_bind_dout [0:0];
  wire deal8_tag_in_bind_full_n [0:0];
  wire deal8_tag_in_bind_write [0:0];
  wire deal8_tag_in_bind_empty_n [0:0];
  wire deal8_tag_in_bind_read [0:0];
  genvar deal8_tag_in_bind_i;
  generate
    for (deal8_tag_in_bind_i = 0; deal8_tag_in_bind_i < 1; deal8_tag_in_bind_i = deal8_tag_in_bind_i + 1) begin : g_deal8_tag_in_bind
      spmw_fifo #(.DW(32), .DEPTH(4)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(deal8_tag_in_bind_din[deal8_tag_in_bind_i]), .full_n(deal8_tag_in_bind_full_n[deal8_tag_in_bind_i]), .write(deal8_tag_in_bind_write[deal8_tag_in_bind_i]), .dout(deal8_tag_in_bind_dout[deal8_tag_in_bind_i]), .empty_n(deal8_tag_in_bind_empty_n[deal8_tag_in_bind_i]), .read(deal8_tag_in_bind_read[deal8_tag_in_bind_i]));
    end
  endgenerate
  // family req8_ins_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] req8_ins_in_bind_din [0:0];
  wire [63:0] req8_ins_in_bind_dout [0:0];
  wire req8_ins_in_bind_full_n [0:0];
  wire req8_ins_in_bind_write [0:0];
  wire req8_ins_in_bind_empty_n [0:0];
  wire req8_ins_in_bind_read [0:0];
  genvar req8_ins_in_bind_i;
  generate
    for (req8_ins_in_bind_i = 0; req8_ins_in_bind_i < 1; req8_ins_in_bind_i = req8_ins_in_bind_i + 1) begin : g_req8_ins_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(req8_ins_in_bind_din[req8_ins_in_bind_i]), .full_n(req8_ins_in_bind_full_n[req8_ins_in_bind_i]), .write(req8_ins_in_bind_write[req8_ins_in_bind_i]), .dout(req8_ins_in_bind_dout[req8_ins_in_bind_i]), .empty_n(req8_ins_in_bind_empty_n[req8_ins_in_bind_i]), .read(req8_ins_in_bind_read[req8_ins_in_bind_i]));
    end
  endgenerate
  // family head8_op_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] head8_op_in_bind_din [0:0];
  wire [63:0] head8_op_in_bind_dout [0:0];
  wire head8_op_in_bind_full_n [0:0];
  wire head8_op_in_bind_write [0:0];
  wire head8_op_in_bind_empty_n [0:0];
  wire head8_op_in_bind_read [0:0];
  genvar head8_op_in_bind_i;
  generate
    for (head8_op_in_bind_i = 0; head8_op_in_bind_i < 1; head8_op_in_bind_i = head8_op_in_bind_i + 1) begin : g_head8_op_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(head8_op_in_bind_din[head8_op_in_bind_i]), .full_n(head8_op_in_bind_full_n[head8_op_in_bind_i]), .write(head8_op_in_bind_write[head8_op_in_bind_i]), .dout(head8_op_in_bind_dout[head8_op_in_bind_i]), .empty_n(head8_op_in_bind_empty_n[head8_op_in_bind_i]), .read(head8_op_in_bind_read[head8_op_in_bind_i]));
    end
  endgenerate
  // family wreq8_y_cmd_bind: 1 channel(s), 64-bit, depth 16
  wire [63:0] wreq8_y_cmd_bind_din [0:0];
  wire [63:0] wreq8_y_cmd_bind_dout [0:0];
  wire wreq8_y_cmd_bind_full_n [0:0];
  wire wreq8_y_cmd_bind_write [0:0];
  wire wreq8_y_cmd_bind_empty_n [0:0];
  wire wreq8_y_cmd_bind_read [0:0];
  genvar wreq8_y_cmd_bind_i;
  generate
    for (wreq8_y_cmd_bind_i = 0; wreq8_y_cmd_bind_i < 1; wreq8_y_cmd_bind_i = wreq8_y_cmd_bind_i + 1) begin : g_wreq8_y_cmd_bind
      spmw_fifo #(.DW(64), .DEPTH(16)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(wreq8_y_cmd_bind_din[wreq8_y_cmd_bind_i]), .full_n(wreq8_y_cmd_bind_full_n[wreq8_y_cmd_bind_i]), .write(wreq8_y_cmd_bind_write[wreq8_y_cmd_bind_i]), .dout(wreq8_y_cmd_bind_dout[wreq8_y_cmd_bind_i]), .empty_n(wreq8_y_cmd_bind_empty_n[wreq8_y_cmd_bind_i]), .read(wreq8_y_cmd_bind_read[wreq8_y_cmd_bind_i]));
    end
  endgenerate
  // family head8_a_in_bind: 1 channel(s), 64-bit, depth 128
  wire [63:0] head8_a_in_bind_din [0:0];
  wire [63:0] head8_a_in_bind_dout [0:0];
  wire head8_a_in_bind_full_n [0:0];
  wire head8_a_in_bind_write [0:0];
  wire head8_a_in_bind_empty_n [0:0];
  wire head8_a_in_bind_read [0:0];
  genvar head8_a_in_bind_i;
  generate
    for (head8_a_in_bind_i = 0; head8_a_in_bind_i < 1; head8_a_in_bind_i = head8_a_in_bind_i + 1) begin : g_head8_a_in_bind
      spmw_fifo_bram #(.DW(64), .DEPTH(128)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(head8_a_in_bind_din[head8_a_in_bind_i]), .full_n(head8_a_in_bind_full_n[head8_a_in_bind_i]), .write(head8_a_in_bind_write[head8_a_in_bind_i]), .dout(head8_a_in_bind_dout[head8_a_in_bind_i]), .empty_n(head8_a_in_bind_empty_n[head8_a_in_bind_i]), .read(head8_a_in_bind_read[head8_a_in_bind_i]));
    end
  endgenerate
  // family head8_w_in_bind: 1 channel(s), 64-bit, depth 256
  wire [63:0] head8_w_in_bind_din [0:0];
  wire [63:0] head8_w_in_bind_dout [0:0];
  wire head8_w_in_bind_full_n [0:0];
  wire head8_w_in_bind_write [0:0];
  wire head8_w_in_bind_empty_n [0:0];
  wire head8_w_in_bind_read [0:0];
  genvar head8_w_in_bind_i;
  generate
    for (head8_w_in_bind_i = 0; head8_w_in_bind_i < 1; head8_w_in_bind_i = head8_w_in_bind_i + 1) begin : g_head8_w_in_bind
      spmw_fifo_bram #(.DW(64), .DEPTH(256)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(head8_w_in_bind_din[head8_w_in_bind_i]), .full_n(head8_w_in_bind_full_n[head8_w_in_bind_i]), .write(head8_w_in_bind_write[head8_w_in_bind_i]), .dout(head8_w_in_bind_dout[head8_w_in_bind_i]), .empty_n(head8_w_in_bind_empty_n[head8_w_in_bind_i]), .read(head8_w_in_bind_read[head8_w_in_bind_i]));
    end
  endgenerate
  // family head8_b_in_bind: 1 channel(s), 64-bit, depth 16
  wire [63:0] head8_b_in_bind_din [0:0];
  wire [63:0] head8_b_in_bind_dout [0:0];
  wire head8_b_in_bind_full_n [0:0];
  wire head8_b_in_bind_write [0:0];
  wire head8_b_in_bind_empty_n [0:0];
  wire head8_b_in_bind_read [0:0];
  genvar head8_b_in_bind_i;
  generate
    for (head8_b_in_bind_i = 0; head8_b_in_bind_i < 1; head8_b_in_bind_i = head8_b_in_bind_i + 1) begin : g_head8_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(16)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(head8_b_in_bind_din[head8_b_in_bind_i]), .full_n(head8_b_in_bind_full_n[head8_b_in_bind_i]), .write(head8_b_in_bind_write[head8_b_in_bind_i]), .dout(head8_b_in_bind_dout[head8_b_in_bind_i]), .empty_n(head8_b_in_bind_empty_n[head8_b_in_bind_i]), .read(head8_b_in_bind_read[head8_b_in_bind_i]));
    end
  endgenerate
  // family uq_u_in_bind: 1 channel(s), 64-bit, depth 32
  wire [63:0] uq_u_in_bind_din [0:0];
  wire [63:0] uq_u_in_bind_dout [0:0];
  wire uq_u_in_bind_full_n [0:0];
  wire uq_u_in_bind_write [0:0];
  wire uq_u_in_bind_empty_n [0:0];
  wire uq_u_in_bind_read [0:0];
  genvar uq_u_in_bind_i;
  generate
    for (uq_u_in_bind_i = 0; uq_u_in_bind_i < 1; uq_u_in_bind_i = uq_u_in_bind_i + 1) begin : g_uq_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(32)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(uq_u_in_bind_din[uq_u_in_bind_i]), .full_n(uq_u_in_bind_full_n[uq_u_in_bind_i]), .write(uq_u_in_bind_write[uq_u_in_bind_i]), .dout(uq_u_in_bind_dout[uq_u_in_bind_i]), .empty_n(uq_u_in_bind_empty_n[uq_u_in_bind_i]), .read(uq_u_in_bind_read[uq_u_in_bind_i]));
    end
  endgenerate
  // family head8_credit_bind: 1 channel(s), 8-bit, depth 1024
  wire [7:0] head8_credit_bind_din [0:0];
  wire [7:0] head8_credit_bind_dout [0:0];
  wire head8_credit_bind_full_n [0:0];
  wire head8_credit_bind_write [0:0];
  wire head8_credit_bind_empty_n [0:0];
  wire head8_credit_bind_read [0:0];
  genvar head8_credit_bind_i;
  generate
    for (head8_credit_bind_i = 0; head8_credit_bind_i < 1; head8_credit_bind_i = head8_credit_bind_i + 1) begin : g_head8_credit_bind
      spmw_fifo_bram #(.DW(8), .DEPTH(1024)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(head8_credit_bind_din[head8_credit_bind_i]), .full_n(head8_credit_bind_full_n[head8_credit_bind_i]), .write(head8_credit_bind_write[head8_credit_bind_i]), .dout(head8_credit_bind_dout[head8_credit_bind_i]), .empty_n(head8_credit_bind_empty_n[head8_credit_bind_i]), .read(head8_credit_bind_read[head8_credit_bind_i]));
    end
  endgenerate
  // family tap_u_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] tap_u_in_bind_din [0:0];
  wire [63:0] tap_u_in_bind_dout [0:0];
  wire tap_u_in_bind_full_n [0:0];
  wire tap_u_in_bind_write [0:0];
  wire tap_u_in_bind_empty_n [0:0];
  wire tap_u_in_bind_read [0:0];
  genvar tap_u_in_bind_i;
  generate
    for (tap_u_in_bind_i = 0; tap_u_in_bind_i < 1; tap_u_in_bind_i = tap_u_in_bind_i + 1) begin : g_tap_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tap_u_in_bind_din[tap_u_in_bind_i]), .full_n(tap_u_in_bind_full_n[tap_u_in_bind_i]), .write(tap_u_in_bind_write[tap_u_in_bind_i]), .dout(tap_u_in_bind_dout[tap_u_in_bind_i]), .empty_n(tap_u_in_bind_empty_n[tap_u_in_bind_i]), .read(tap_u_in_bind_read[tap_u_in_bind_i]));
    end
  endgenerate
  // family tap_u_out_u_in: 8 channel(s), 64-bit, depth 2
  wire [63:0] tap_u_out_u_in_din [0:7];
  wire [63:0] tap_u_out_u_in_dout [0:7];
  wire tap_u_out_u_in_full_n [0:7];
  wire tap_u_out_u_in_write [0:7];
  wire tap_u_out_u_in_empty_n [0:7];
  wire tap_u_out_u_in_read [0:7];
  genvar tap_u_out_u_in_i;
  generate
    for (tap_u_out_u_in_i = 0; tap_u_out_u_in_i < 8; tap_u_out_u_in_i = tap_u_out_u_in_i + 1) begin : g_tap_u_out_u_in
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tap_u_out_u_in_din[tap_u_out_u_in_i]), .full_n(tap_u_out_u_in_full_n[tap_u_out_u_in_i]), .write(tap_u_out_u_in_write[tap_u_out_u_in_i]), .dout(tap_u_out_u_in_dout[tap_u_out_u_in_i]), .empty_n(tap_u_out_u_in_empty_n[tap_u_out_u_in_i]), .read(tap_u_out_u_in_read[tap_u_out_u_in_i]));
    end
  endgenerate
  // family lane_c_in_bind: 8 channel(s), 64-bit, depth 2
  wire [63:0] lane_c_in_bind_din [0:7];
  wire [63:0] lane_c_in_bind_dout [0:7];
  wire lane_c_in_bind_full_n [0:7];
  wire lane_c_in_bind_write [0:7];
  wire lane_c_in_bind_empty_n [0:7];
  wire lane_c_in_bind_read [0:7];
  genvar lane_c_in_bind_i;
  generate
    for (lane_c_in_bind_i = 0; lane_c_in_bind_i < 8; lane_c_in_bind_i = lane_c_in_bind_i + 1) begin : g_lane_c_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(lane_c_in_bind_din[lane_c_in_bind_i]), .full_n(lane_c_in_bind_full_n[lane_c_in_bind_i]), .write(lane_c_in_bind_write[lane_c_in_bind_i]), .dout(lane_c_in_bind_dout[lane_c_in_bind_i]), .empty_n(lane_c_in_bind_empty_n[lane_c_in_bind_i]), .read(lane_c_in_bind_read[lane_c_in_bind_i]));
    end
  endgenerate
  // family ctap_y_in_bind: 8 channel(s), 16-bit, depth 2
  wire [15:0] ctap_y_in_bind_din [0:7];
  wire [15:0] ctap_y_in_bind_dout [0:7];
  wire ctap_y_in_bind_full_n [0:7];
  wire ctap_y_in_bind_write [0:7];
  wire ctap_y_in_bind_empty_n [0:7];
  wire ctap_y_in_bind_read [0:7];
  genvar ctap_y_in_bind_i;
  generate
    for (ctap_y_in_bind_i = 0; ctap_y_in_bind_i < 8; ctap_y_in_bind_i = ctap_y_in_bind_i + 1) begin : g_ctap_y_in_bind
      spmw_fifo #(.DW(16), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(ctap_y_in_bind_din[ctap_y_in_bind_i]), .full_n(ctap_y_in_bind_full_n[ctap_y_in_bind_i]), .write(ctap_y_in_bind_write[ctap_y_in_bind_i]), .dout(ctap_y_in_bind_dout[ctap_y_in_bind_i]), .empty_n(ctap_y_in_bind_empty_n[ctap_y_in_bind_i]), .read(ctap_y_in_bind_read[ctap_y_in_bind_i]));
    end
  endgenerate
  // family ctap_r_out_r_in: 8 channel(s), 72-bit, depth 2
  wire [71:0] ctap_r_out_r_in_din [0:7];
  wire [71:0] ctap_r_out_r_in_dout [0:7];
  wire ctap_r_out_r_in_full_n [0:7];
  wire ctap_r_out_r_in_write [0:7];
  wire ctap_r_out_r_in_empty_n [0:7];
  wire ctap_r_out_r_in_read [0:7];
  genvar ctap_r_out_r_in_i;
  generate
    for (ctap_r_out_r_in_i = 0; ctap_r_out_r_in_i < 8; ctap_r_out_r_in_i = ctap_r_out_r_in_i + 1) begin : g_ctap_r_out_r_in
      spmw_fifo #(.DW(72), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(ctap_r_out_r_in_din[ctap_r_out_r_in_i]), .full_n(ctap_r_out_r_in_full_n[ctap_r_out_r_in_i]), .write(ctap_r_out_r_in_write[ctap_r_out_r_in_i]), .dout(ctap_r_out_r_in_dout[ctap_r_out_r_in_i]), .empty_n(ctap_r_out_r_in_empty_n[ctap_r_out_r_in_i]), .read(ctap_r_out_r_in_read[ctap_r_out_r_in_i]));
    end
  endgenerate
  // family pack8_row_in_bind: 1 channel(s), 72-bit, depth 1024
  wire [71:0] pack8_row_in_bind_din [0:0];
  wire [71:0] pack8_row_in_bind_dout [0:0];
  wire pack8_row_in_bind_full_n [0:0];
  wire pack8_row_in_bind_write [0:0];
  wire pack8_row_in_bind_empty_n [0:0];
  wire pack8_row_in_bind_read [0:0];
  genvar pack8_row_in_bind_i;
  generate
    for (pack8_row_in_bind_i = 0; pack8_row_in_bind_i < 1; pack8_row_in_bind_i = pack8_row_in_bind_i + 1) begin : g_pack8_row_in_bind
      spmw_fifo_bram #(.DW(72), .DEPTH(1024)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pack8_row_in_bind_din[pack8_row_in_bind_i]), .full_n(pack8_row_in_bind_full_n[pack8_row_in_bind_i]), .write(pack8_row_in_bind_write[pack8_row_in_bind_i]), .dout(pack8_row_in_bind_dout[pack8_row_in_bind_i]), .empty_n(pack8_row_in_bind_empty_n[pack8_row_in_bind_i]), .read(pack8_row_in_bind_read[pack8_row_in_bind_i]));
    end
  endgenerate
  // role pe_r0: 36 instance(s)
  pe_r0 u_pe_r0_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[9]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[9]),
      .a_in_read(pe_a_out_a_in_read[9]),
      .a_out_din(pe_a_out_a_in_din[10]),
      .a_out_full_n(pe_a_out_a_in_full_n[10]),
      .a_out_write(pe_a_out_a_in_write[10]),
      .p_in_dout(pe_p_out_p_in_dout[9]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[9]),
      .p_in_read(pe_p_out_p_in_read[9]),
      .p_out_din(pe_p_out_p_in_din[17]),
      .p_out_full_n(pe_p_out_p_in_full_n[17]),
      .p_out_write(pe_p_out_p_in_write[17]),
      .w_in_dout(pe_w_out_w_in_dout[9]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[9]),
      .w_in_read(pe_w_out_w_in_read[9]),
      .w_out_din(pe_w_out_w_in_din[10]),
      .w_out_full_n(pe_w_out_w_in_full_n[10]),
      .w_out_write(pe_w_out_w_in_write[10]));
  pe_r0 u_pe_r0_1_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[10]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[10]),
      .a_in_read(pe_a_out_a_in_read[10]),
      .a_out_din(pe_a_out_a_in_din[11]),
      .a_out_full_n(pe_a_out_a_in_full_n[11]),
      .a_out_write(pe_a_out_a_in_write[11]),
      .p_in_dout(pe_p_out_p_in_dout[10]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[10]),
      .p_in_read(pe_p_out_p_in_read[10]),
      .p_out_din(pe_p_out_p_in_din[18]),
      .p_out_full_n(pe_p_out_p_in_full_n[18]),
      .p_out_write(pe_p_out_p_in_write[18]),
      .w_in_dout(pe_w_out_w_in_dout[10]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[10]),
      .w_in_read(pe_w_out_w_in_read[10]),
      .w_out_din(pe_w_out_w_in_din[11]),
      .w_out_full_n(pe_w_out_w_in_full_n[11]),
      .w_out_write(pe_w_out_w_in_write[11]));
  pe_r0 u_pe_r0_1_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[11]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[11]),
      .a_in_read(pe_a_out_a_in_read[11]),
      .a_out_din(pe_a_out_a_in_din[12]),
      .a_out_full_n(pe_a_out_a_in_full_n[12]),
      .a_out_write(pe_a_out_a_in_write[12]),
      .p_in_dout(pe_p_out_p_in_dout[11]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[11]),
      .p_in_read(pe_p_out_p_in_read[11]),
      .p_out_din(pe_p_out_p_in_din[19]),
      .p_out_full_n(pe_p_out_p_in_full_n[19]),
      .p_out_write(pe_p_out_p_in_write[19]),
      .w_in_dout(pe_w_out_w_in_dout[11]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[11]),
      .w_in_read(pe_w_out_w_in_read[11]),
      .w_out_din(pe_w_out_w_in_din[12]),
      .w_out_full_n(pe_w_out_w_in_full_n[12]),
      .w_out_write(pe_w_out_w_in_write[12]));
  pe_r0 u_pe_r0_1_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[12]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[12]),
      .a_in_read(pe_a_out_a_in_read[12]),
      .a_out_din(pe_a_out_a_in_din[13]),
      .a_out_full_n(pe_a_out_a_in_full_n[13]),
      .a_out_write(pe_a_out_a_in_write[13]),
      .p_in_dout(pe_p_out_p_in_dout[12]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[12]),
      .p_in_read(pe_p_out_p_in_read[12]),
      .p_out_din(pe_p_out_p_in_din[20]),
      .p_out_full_n(pe_p_out_p_in_full_n[20]),
      .p_out_write(pe_p_out_p_in_write[20]),
      .w_in_dout(pe_w_out_w_in_dout[12]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[12]),
      .w_in_read(pe_w_out_w_in_read[12]),
      .w_out_din(pe_w_out_w_in_din[13]),
      .w_out_full_n(pe_w_out_w_in_full_n[13]),
      .w_out_write(pe_w_out_w_in_write[13]));
  pe_r0 u_pe_r0_1_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[13]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[13]),
      .a_in_read(pe_a_out_a_in_read[13]),
      .a_out_din(pe_a_out_a_in_din[14]),
      .a_out_full_n(pe_a_out_a_in_full_n[14]),
      .a_out_write(pe_a_out_a_in_write[14]),
      .p_in_dout(pe_p_out_p_in_dout[13]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[13]),
      .p_in_read(pe_p_out_p_in_read[13]),
      .p_out_din(pe_p_out_p_in_din[21]),
      .p_out_full_n(pe_p_out_p_in_full_n[21]),
      .p_out_write(pe_p_out_p_in_write[21]),
      .w_in_dout(pe_w_out_w_in_dout[13]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[13]),
      .w_in_read(pe_w_out_w_in_read[13]),
      .w_out_din(pe_w_out_w_in_din[14]),
      .w_out_full_n(pe_w_out_w_in_full_n[14]),
      .w_out_write(pe_w_out_w_in_write[14]));
  pe_r0 u_pe_r0_1_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[14]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[14]),
      .a_in_read(pe_a_out_a_in_read[14]),
      .a_out_din(pe_a_out_a_in_din[15]),
      .a_out_full_n(pe_a_out_a_in_full_n[15]),
      .a_out_write(pe_a_out_a_in_write[15]),
      .p_in_dout(pe_p_out_p_in_dout[14]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[14]),
      .p_in_read(pe_p_out_p_in_read[14]),
      .p_out_din(pe_p_out_p_in_din[22]),
      .p_out_full_n(pe_p_out_p_in_full_n[22]),
      .p_out_write(pe_p_out_p_in_write[22]),
      .w_in_dout(pe_w_out_w_in_dout[14]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[14]),
      .w_in_read(pe_w_out_w_in_read[14]),
      .w_out_din(pe_w_out_w_in_din[15]),
      .w_out_full_n(pe_w_out_w_in_full_n[15]),
      .w_out_write(pe_w_out_w_in_write[15]));
  pe_r0 u_pe_r0_2_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[17]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[17]),
      .a_in_read(pe_a_out_a_in_read[17]),
      .a_out_din(pe_a_out_a_in_din[18]),
      .a_out_full_n(pe_a_out_a_in_full_n[18]),
      .a_out_write(pe_a_out_a_in_write[18]),
      .p_in_dout(pe_p_out_p_in_dout[17]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[17]),
      .p_in_read(pe_p_out_p_in_read[17]),
      .p_out_din(pe_p_out_p_in_din[25]),
      .p_out_full_n(pe_p_out_p_in_full_n[25]),
      .p_out_write(pe_p_out_p_in_write[25]),
      .w_in_dout(pe_w_out_w_in_dout[17]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[17]),
      .w_in_read(pe_w_out_w_in_read[17]),
      .w_out_din(pe_w_out_w_in_din[18]),
      .w_out_full_n(pe_w_out_w_in_full_n[18]),
      .w_out_write(pe_w_out_w_in_write[18]));
  pe_r0 u_pe_r0_2_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[18]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[18]),
      .a_in_read(pe_a_out_a_in_read[18]),
      .a_out_din(pe_a_out_a_in_din[19]),
      .a_out_full_n(pe_a_out_a_in_full_n[19]),
      .a_out_write(pe_a_out_a_in_write[19]),
      .p_in_dout(pe_p_out_p_in_dout[18]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[18]),
      .p_in_read(pe_p_out_p_in_read[18]),
      .p_out_din(pe_p_out_p_in_din[26]),
      .p_out_full_n(pe_p_out_p_in_full_n[26]),
      .p_out_write(pe_p_out_p_in_write[26]),
      .w_in_dout(pe_w_out_w_in_dout[18]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[18]),
      .w_in_read(pe_w_out_w_in_read[18]),
      .w_out_din(pe_w_out_w_in_din[19]),
      .w_out_full_n(pe_w_out_w_in_full_n[19]),
      .w_out_write(pe_w_out_w_in_write[19]));
  pe_r0 u_pe_r0_2_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[19]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[19]),
      .a_in_read(pe_a_out_a_in_read[19]),
      .a_out_din(pe_a_out_a_in_din[20]),
      .a_out_full_n(pe_a_out_a_in_full_n[20]),
      .a_out_write(pe_a_out_a_in_write[20]),
      .p_in_dout(pe_p_out_p_in_dout[19]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[19]),
      .p_in_read(pe_p_out_p_in_read[19]),
      .p_out_din(pe_p_out_p_in_din[27]),
      .p_out_full_n(pe_p_out_p_in_full_n[27]),
      .p_out_write(pe_p_out_p_in_write[27]),
      .w_in_dout(pe_w_out_w_in_dout[19]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[19]),
      .w_in_read(pe_w_out_w_in_read[19]),
      .w_out_din(pe_w_out_w_in_din[20]),
      .w_out_full_n(pe_w_out_w_in_full_n[20]),
      .w_out_write(pe_w_out_w_in_write[20]));
  pe_r0 u_pe_r0_2_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[20]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[20]),
      .a_in_read(pe_a_out_a_in_read[20]),
      .a_out_din(pe_a_out_a_in_din[21]),
      .a_out_full_n(pe_a_out_a_in_full_n[21]),
      .a_out_write(pe_a_out_a_in_write[21]),
      .p_in_dout(pe_p_out_p_in_dout[20]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[20]),
      .p_in_read(pe_p_out_p_in_read[20]),
      .p_out_din(pe_p_out_p_in_din[28]),
      .p_out_full_n(pe_p_out_p_in_full_n[28]),
      .p_out_write(pe_p_out_p_in_write[28]),
      .w_in_dout(pe_w_out_w_in_dout[20]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[20]),
      .w_in_read(pe_w_out_w_in_read[20]),
      .w_out_din(pe_w_out_w_in_din[21]),
      .w_out_full_n(pe_w_out_w_in_full_n[21]),
      .w_out_write(pe_w_out_w_in_write[21]));
  pe_r0 u_pe_r0_2_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[21]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[21]),
      .a_in_read(pe_a_out_a_in_read[21]),
      .a_out_din(pe_a_out_a_in_din[22]),
      .a_out_full_n(pe_a_out_a_in_full_n[22]),
      .a_out_write(pe_a_out_a_in_write[22]),
      .p_in_dout(pe_p_out_p_in_dout[21]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[21]),
      .p_in_read(pe_p_out_p_in_read[21]),
      .p_out_din(pe_p_out_p_in_din[29]),
      .p_out_full_n(pe_p_out_p_in_full_n[29]),
      .p_out_write(pe_p_out_p_in_write[29]),
      .w_in_dout(pe_w_out_w_in_dout[21]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[21]),
      .w_in_read(pe_w_out_w_in_read[21]),
      .w_out_din(pe_w_out_w_in_din[22]),
      .w_out_full_n(pe_w_out_w_in_full_n[22]),
      .w_out_write(pe_w_out_w_in_write[22]));
  pe_r0 u_pe_r0_2_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[22]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[22]),
      .a_in_read(pe_a_out_a_in_read[22]),
      .a_out_din(pe_a_out_a_in_din[23]),
      .a_out_full_n(pe_a_out_a_in_full_n[23]),
      .a_out_write(pe_a_out_a_in_write[23]),
      .p_in_dout(pe_p_out_p_in_dout[22]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[22]),
      .p_in_read(pe_p_out_p_in_read[22]),
      .p_out_din(pe_p_out_p_in_din[30]),
      .p_out_full_n(pe_p_out_p_in_full_n[30]),
      .p_out_write(pe_p_out_p_in_write[30]),
      .w_in_dout(pe_w_out_w_in_dout[22]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[22]),
      .w_in_read(pe_w_out_w_in_read[22]),
      .w_out_din(pe_w_out_w_in_din[23]),
      .w_out_full_n(pe_w_out_w_in_full_n[23]),
      .w_out_write(pe_w_out_w_in_write[23]));
  pe_r0 u_pe_r0_3_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[25]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[25]),
      .a_in_read(pe_a_out_a_in_read[25]),
      .a_out_din(pe_a_out_a_in_din[26]),
      .a_out_full_n(pe_a_out_a_in_full_n[26]),
      .a_out_write(pe_a_out_a_in_write[26]),
      .p_in_dout(pe_p_out_p_in_dout[25]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[25]),
      .p_in_read(pe_p_out_p_in_read[25]),
      .p_out_din(pe_p_out_p_in_din[33]),
      .p_out_full_n(pe_p_out_p_in_full_n[33]),
      .p_out_write(pe_p_out_p_in_write[33]),
      .w_in_dout(pe_w_out_w_in_dout[25]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[25]),
      .w_in_read(pe_w_out_w_in_read[25]),
      .w_out_din(pe_w_out_w_in_din[26]),
      .w_out_full_n(pe_w_out_w_in_full_n[26]),
      .w_out_write(pe_w_out_w_in_write[26]));
  pe_r0 u_pe_r0_3_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[26]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[26]),
      .a_in_read(pe_a_out_a_in_read[26]),
      .a_out_din(pe_a_out_a_in_din[27]),
      .a_out_full_n(pe_a_out_a_in_full_n[27]),
      .a_out_write(pe_a_out_a_in_write[27]),
      .p_in_dout(pe_p_out_p_in_dout[26]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[26]),
      .p_in_read(pe_p_out_p_in_read[26]),
      .p_out_din(pe_p_out_p_in_din[34]),
      .p_out_full_n(pe_p_out_p_in_full_n[34]),
      .p_out_write(pe_p_out_p_in_write[34]),
      .w_in_dout(pe_w_out_w_in_dout[26]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[26]),
      .w_in_read(pe_w_out_w_in_read[26]),
      .w_out_din(pe_w_out_w_in_din[27]),
      .w_out_full_n(pe_w_out_w_in_full_n[27]),
      .w_out_write(pe_w_out_w_in_write[27]));
  pe_r0 u_pe_r0_3_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[27]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[27]),
      .a_in_read(pe_a_out_a_in_read[27]),
      .a_out_din(pe_a_out_a_in_din[28]),
      .a_out_full_n(pe_a_out_a_in_full_n[28]),
      .a_out_write(pe_a_out_a_in_write[28]),
      .p_in_dout(pe_p_out_p_in_dout[27]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[27]),
      .p_in_read(pe_p_out_p_in_read[27]),
      .p_out_din(pe_p_out_p_in_din[35]),
      .p_out_full_n(pe_p_out_p_in_full_n[35]),
      .p_out_write(pe_p_out_p_in_write[35]),
      .w_in_dout(pe_w_out_w_in_dout[27]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[27]),
      .w_in_read(pe_w_out_w_in_read[27]),
      .w_out_din(pe_w_out_w_in_din[28]),
      .w_out_full_n(pe_w_out_w_in_full_n[28]),
      .w_out_write(pe_w_out_w_in_write[28]));
  pe_r0 u_pe_r0_3_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[28]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[28]),
      .a_in_read(pe_a_out_a_in_read[28]),
      .a_out_din(pe_a_out_a_in_din[29]),
      .a_out_full_n(pe_a_out_a_in_full_n[29]),
      .a_out_write(pe_a_out_a_in_write[29]),
      .p_in_dout(pe_p_out_p_in_dout[28]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[28]),
      .p_in_read(pe_p_out_p_in_read[28]),
      .p_out_din(pe_p_out_p_in_din[36]),
      .p_out_full_n(pe_p_out_p_in_full_n[36]),
      .p_out_write(pe_p_out_p_in_write[36]),
      .w_in_dout(pe_w_out_w_in_dout[28]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[28]),
      .w_in_read(pe_w_out_w_in_read[28]),
      .w_out_din(pe_w_out_w_in_din[29]),
      .w_out_full_n(pe_w_out_w_in_full_n[29]),
      .w_out_write(pe_w_out_w_in_write[29]));
  pe_r0 u_pe_r0_3_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[29]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[29]),
      .a_in_read(pe_a_out_a_in_read[29]),
      .a_out_din(pe_a_out_a_in_din[30]),
      .a_out_full_n(pe_a_out_a_in_full_n[30]),
      .a_out_write(pe_a_out_a_in_write[30]),
      .p_in_dout(pe_p_out_p_in_dout[29]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[29]),
      .p_in_read(pe_p_out_p_in_read[29]),
      .p_out_din(pe_p_out_p_in_din[37]),
      .p_out_full_n(pe_p_out_p_in_full_n[37]),
      .p_out_write(pe_p_out_p_in_write[37]),
      .w_in_dout(pe_w_out_w_in_dout[29]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[29]),
      .w_in_read(pe_w_out_w_in_read[29]),
      .w_out_din(pe_w_out_w_in_din[30]),
      .w_out_full_n(pe_w_out_w_in_full_n[30]),
      .w_out_write(pe_w_out_w_in_write[30]));
  pe_r0 u_pe_r0_3_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[30]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[30]),
      .a_in_read(pe_a_out_a_in_read[30]),
      .a_out_din(pe_a_out_a_in_din[31]),
      .a_out_full_n(pe_a_out_a_in_full_n[31]),
      .a_out_write(pe_a_out_a_in_write[31]),
      .p_in_dout(pe_p_out_p_in_dout[30]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[30]),
      .p_in_read(pe_p_out_p_in_read[30]),
      .p_out_din(pe_p_out_p_in_din[38]),
      .p_out_full_n(pe_p_out_p_in_full_n[38]),
      .p_out_write(pe_p_out_p_in_write[38]),
      .w_in_dout(pe_w_out_w_in_dout[30]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[30]),
      .w_in_read(pe_w_out_w_in_read[30]),
      .w_out_din(pe_w_out_w_in_din[31]),
      .w_out_full_n(pe_w_out_w_in_full_n[31]),
      .w_out_write(pe_w_out_w_in_write[31]));
  pe_r0 u_pe_r0_4_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[33]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[33]),
      .a_in_read(pe_a_out_a_in_read[33]),
      .a_out_din(pe_a_out_a_in_din[34]),
      .a_out_full_n(pe_a_out_a_in_full_n[34]),
      .a_out_write(pe_a_out_a_in_write[34]),
      .p_in_dout(pe_p_out_p_in_dout[33]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[33]),
      .p_in_read(pe_p_out_p_in_read[33]),
      .p_out_din(pe_p_out_p_in_din[41]),
      .p_out_full_n(pe_p_out_p_in_full_n[41]),
      .p_out_write(pe_p_out_p_in_write[41]),
      .w_in_dout(pe_w_out_w_in_dout[33]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[33]),
      .w_in_read(pe_w_out_w_in_read[33]),
      .w_out_din(pe_w_out_w_in_din[34]),
      .w_out_full_n(pe_w_out_w_in_full_n[34]),
      .w_out_write(pe_w_out_w_in_write[34]));
  pe_r0 u_pe_r0_4_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[34]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[34]),
      .a_in_read(pe_a_out_a_in_read[34]),
      .a_out_din(pe_a_out_a_in_din[35]),
      .a_out_full_n(pe_a_out_a_in_full_n[35]),
      .a_out_write(pe_a_out_a_in_write[35]),
      .p_in_dout(pe_p_out_p_in_dout[34]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[34]),
      .p_in_read(pe_p_out_p_in_read[34]),
      .p_out_din(pe_p_out_p_in_din[42]),
      .p_out_full_n(pe_p_out_p_in_full_n[42]),
      .p_out_write(pe_p_out_p_in_write[42]),
      .w_in_dout(pe_w_out_w_in_dout[34]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[34]),
      .w_in_read(pe_w_out_w_in_read[34]),
      .w_out_din(pe_w_out_w_in_din[35]),
      .w_out_full_n(pe_w_out_w_in_full_n[35]),
      .w_out_write(pe_w_out_w_in_write[35]));
  pe_r0 u_pe_r0_4_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[35]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[35]),
      .a_in_read(pe_a_out_a_in_read[35]),
      .a_out_din(pe_a_out_a_in_din[36]),
      .a_out_full_n(pe_a_out_a_in_full_n[36]),
      .a_out_write(pe_a_out_a_in_write[36]),
      .p_in_dout(pe_p_out_p_in_dout[35]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[35]),
      .p_in_read(pe_p_out_p_in_read[35]),
      .p_out_din(pe_p_out_p_in_din[43]),
      .p_out_full_n(pe_p_out_p_in_full_n[43]),
      .p_out_write(pe_p_out_p_in_write[43]),
      .w_in_dout(pe_w_out_w_in_dout[35]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[35]),
      .w_in_read(pe_w_out_w_in_read[35]),
      .w_out_din(pe_w_out_w_in_din[36]),
      .w_out_full_n(pe_w_out_w_in_full_n[36]),
      .w_out_write(pe_w_out_w_in_write[36]));
  pe_r0 u_pe_r0_4_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[36]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[36]),
      .a_in_read(pe_a_out_a_in_read[36]),
      .a_out_din(pe_a_out_a_in_din[37]),
      .a_out_full_n(pe_a_out_a_in_full_n[37]),
      .a_out_write(pe_a_out_a_in_write[37]),
      .p_in_dout(pe_p_out_p_in_dout[36]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[36]),
      .p_in_read(pe_p_out_p_in_read[36]),
      .p_out_din(pe_p_out_p_in_din[44]),
      .p_out_full_n(pe_p_out_p_in_full_n[44]),
      .p_out_write(pe_p_out_p_in_write[44]),
      .w_in_dout(pe_w_out_w_in_dout[36]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[36]),
      .w_in_read(pe_w_out_w_in_read[36]),
      .w_out_din(pe_w_out_w_in_din[37]),
      .w_out_full_n(pe_w_out_w_in_full_n[37]),
      .w_out_write(pe_w_out_w_in_write[37]));
  pe_r0 u_pe_r0_4_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[37]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[37]),
      .a_in_read(pe_a_out_a_in_read[37]),
      .a_out_din(pe_a_out_a_in_din[38]),
      .a_out_full_n(pe_a_out_a_in_full_n[38]),
      .a_out_write(pe_a_out_a_in_write[38]),
      .p_in_dout(pe_p_out_p_in_dout[37]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[37]),
      .p_in_read(pe_p_out_p_in_read[37]),
      .p_out_din(pe_p_out_p_in_din[45]),
      .p_out_full_n(pe_p_out_p_in_full_n[45]),
      .p_out_write(pe_p_out_p_in_write[45]),
      .w_in_dout(pe_w_out_w_in_dout[37]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[37]),
      .w_in_read(pe_w_out_w_in_read[37]),
      .w_out_din(pe_w_out_w_in_din[38]),
      .w_out_full_n(pe_w_out_w_in_full_n[38]),
      .w_out_write(pe_w_out_w_in_write[38]));
  pe_r0 u_pe_r0_4_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[38]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[38]),
      .a_in_read(pe_a_out_a_in_read[38]),
      .a_out_din(pe_a_out_a_in_din[39]),
      .a_out_full_n(pe_a_out_a_in_full_n[39]),
      .a_out_write(pe_a_out_a_in_write[39]),
      .p_in_dout(pe_p_out_p_in_dout[38]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[38]),
      .p_in_read(pe_p_out_p_in_read[38]),
      .p_out_din(pe_p_out_p_in_din[46]),
      .p_out_full_n(pe_p_out_p_in_full_n[46]),
      .p_out_write(pe_p_out_p_in_write[46]),
      .w_in_dout(pe_w_out_w_in_dout[38]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[38]),
      .w_in_read(pe_w_out_w_in_read[38]),
      .w_out_din(pe_w_out_w_in_din[39]),
      .w_out_full_n(pe_w_out_w_in_full_n[39]),
      .w_out_write(pe_w_out_w_in_write[39]));
  pe_r0 u_pe_r0_5_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[41]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[41]),
      .a_in_read(pe_a_out_a_in_read[41]),
      .a_out_din(pe_a_out_a_in_din[42]),
      .a_out_full_n(pe_a_out_a_in_full_n[42]),
      .a_out_write(pe_a_out_a_in_write[42]),
      .p_in_dout(pe_p_out_p_in_dout[41]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[41]),
      .p_in_read(pe_p_out_p_in_read[41]),
      .p_out_din(pe_p_out_p_in_din[49]),
      .p_out_full_n(pe_p_out_p_in_full_n[49]),
      .p_out_write(pe_p_out_p_in_write[49]),
      .w_in_dout(pe_w_out_w_in_dout[41]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[41]),
      .w_in_read(pe_w_out_w_in_read[41]),
      .w_out_din(pe_w_out_w_in_din[42]),
      .w_out_full_n(pe_w_out_w_in_full_n[42]),
      .w_out_write(pe_w_out_w_in_write[42]));
  pe_r0 u_pe_r0_5_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[42]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[42]),
      .a_in_read(pe_a_out_a_in_read[42]),
      .a_out_din(pe_a_out_a_in_din[43]),
      .a_out_full_n(pe_a_out_a_in_full_n[43]),
      .a_out_write(pe_a_out_a_in_write[43]),
      .p_in_dout(pe_p_out_p_in_dout[42]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[42]),
      .p_in_read(pe_p_out_p_in_read[42]),
      .p_out_din(pe_p_out_p_in_din[50]),
      .p_out_full_n(pe_p_out_p_in_full_n[50]),
      .p_out_write(pe_p_out_p_in_write[50]),
      .w_in_dout(pe_w_out_w_in_dout[42]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[42]),
      .w_in_read(pe_w_out_w_in_read[42]),
      .w_out_din(pe_w_out_w_in_din[43]),
      .w_out_full_n(pe_w_out_w_in_full_n[43]),
      .w_out_write(pe_w_out_w_in_write[43]));
  pe_r0 u_pe_r0_5_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[43]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[43]),
      .a_in_read(pe_a_out_a_in_read[43]),
      .a_out_din(pe_a_out_a_in_din[44]),
      .a_out_full_n(pe_a_out_a_in_full_n[44]),
      .a_out_write(pe_a_out_a_in_write[44]),
      .p_in_dout(pe_p_out_p_in_dout[43]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[43]),
      .p_in_read(pe_p_out_p_in_read[43]),
      .p_out_din(pe_p_out_p_in_din[51]),
      .p_out_full_n(pe_p_out_p_in_full_n[51]),
      .p_out_write(pe_p_out_p_in_write[51]),
      .w_in_dout(pe_w_out_w_in_dout[43]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[43]),
      .w_in_read(pe_w_out_w_in_read[43]),
      .w_out_din(pe_w_out_w_in_din[44]),
      .w_out_full_n(pe_w_out_w_in_full_n[44]),
      .w_out_write(pe_w_out_w_in_write[44]));
  pe_r0 u_pe_r0_5_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[44]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[44]),
      .a_in_read(pe_a_out_a_in_read[44]),
      .a_out_din(pe_a_out_a_in_din[45]),
      .a_out_full_n(pe_a_out_a_in_full_n[45]),
      .a_out_write(pe_a_out_a_in_write[45]),
      .p_in_dout(pe_p_out_p_in_dout[44]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[44]),
      .p_in_read(pe_p_out_p_in_read[44]),
      .p_out_din(pe_p_out_p_in_din[52]),
      .p_out_full_n(pe_p_out_p_in_full_n[52]),
      .p_out_write(pe_p_out_p_in_write[52]),
      .w_in_dout(pe_w_out_w_in_dout[44]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[44]),
      .w_in_read(pe_w_out_w_in_read[44]),
      .w_out_din(pe_w_out_w_in_din[45]),
      .w_out_full_n(pe_w_out_w_in_full_n[45]),
      .w_out_write(pe_w_out_w_in_write[45]));
  pe_r0 u_pe_r0_5_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[45]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[45]),
      .a_in_read(pe_a_out_a_in_read[45]),
      .a_out_din(pe_a_out_a_in_din[46]),
      .a_out_full_n(pe_a_out_a_in_full_n[46]),
      .a_out_write(pe_a_out_a_in_write[46]),
      .p_in_dout(pe_p_out_p_in_dout[45]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[45]),
      .p_in_read(pe_p_out_p_in_read[45]),
      .p_out_din(pe_p_out_p_in_din[53]),
      .p_out_full_n(pe_p_out_p_in_full_n[53]),
      .p_out_write(pe_p_out_p_in_write[53]),
      .w_in_dout(pe_w_out_w_in_dout[45]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[45]),
      .w_in_read(pe_w_out_w_in_read[45]),
      .w_out_din(pe_w_out_w_in_din[46]),
      .w_out_full_n(pe_w_out_w_in_full_n[46]),
      .w_out_write(pe_w_out_w_in_write[46]));
  pe_r0 u_pe_r0_5_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[46]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[46]),
      .a_in_read(pe_a_out_a_in_read[46]),
      .a_out_din(pe_a_out_a_in_din[47]),
      .a_out_full_n(pe_a_out_a_in_full_n[47]),
      .a_out_write(pe_a_out_a_in_write[47]),
      .p_in_dout(pe_p_out_p_in_dout[46]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[46]),
      .p_in_read(pe_p_out_p_in_read[46]),
      .p_out_din(pe_p_out_p_in_din[54]),
      .p_out_full_n(pe_p_out_p_in_full_n[54]),
      .p_out_write(pe_p_out_p_in_write[54]),
      .w_in_dout(pe_w_out_w_in_dout[46]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[46]),
      .w_in_read(pe_w_out_w_in_read[46]),
      .w_out_din(pe_w_out_w_in_din[47]),
      .w_out_full_n(pe_w_out_w_in_full_n[47]),
      .w_out_write(pe_w_out_w_in_write[47]));
  pe_r0 u_pe_r0_6_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[49]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[49]),
      .a_in_read(pe_a_out_a_in_read[49]),
      .a_out_din(pe_a_out_a_in_din[50]),
      .a_out_full_n(pe_a_out_a_in_full_n[50]),
      .a_out_write(pe_a_out_a_in_write[50]),
      .p_in_dout(pe_p_out_p_in_dout[49]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[49]),
      .p_in_read(pe_p_out_p_in_read[49]),
      .p_out_din(pe_p_out_p_in_din[57]),
      .p_out_full_n(pe_p_out_p_in_full_n[57]),
      .p_out_write(pe_p_out_p_in_write[57]),
      .w_in_dout(pe_w_out_w_in_dout[49]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[49]),
      .w_in_read(pe_w_out_w_in_read[49]),
      .w_out_din(pe_w_out_w_in_din[50]),
      .w_out_full_n(pe_w_out_w_in_full_n[50]),
      .w_out_write(pe_w_out_w_in_write[50]));
  pe_r0 u_pe_r0_6_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[50]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[50]),
      .a_in_read(pe_a_out_a_in_read[50]),
      .a_out_din(pe_a_out_a_in_din[51]),
      .a_out_full_n(pe_a_out_a_in_full_n[51]),
      .a_out_write(pe_a_out_a_in_write[51]),
      .p_in_dout(pe_p_out_p_in_dout[50]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[50]),
      .p_in_read(pe_p_out_p_in_read[50]),
      .p_out_din(pe_p_out_p_in_din[58]),
      .p_out_full_n(pe_p_out_p_in_full_n[58]),
      .p_out_write(pe_p_out_p_in_write[58]),
      .w_in_dout(pe_w_out_w_in_dout[50]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[50]),
      .w_in_read(pe_w_out_w_in_read[50]),
      .w_out_din(pe_w_out_w_in_din[51]),
      .w_out_full_n(pe_w_out_w_in_full_n[51]),
      .w_out_write(pe_w_out_w_in_write[51]));
  pe_r0 u_pe_r0_6_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[51]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[51]),
      .a_in_read(pe_a_out_a_in_read[51]),
      .a_out_din(pe_a_out_a_in_din[52]),
      .a_out_full_n(pe_a_out_a_in_full_n[52]),
      .a_out_write(pe_a_out_a_in_write[52]),
      .p_in_dout(pe_p_out_p_in_dout[51]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[51]),
      .p_in_read(pe_p_out_p_in_read[51]),
      .p_out_din(pe_p_out_p_in_din[59]),
      .p_out_full_n(pe_p_out_p_in_full_n[59]),
      .p_out_write(pe_p_out_p_in_write[59]),
      .w_in_dout(pe_w_out_w_in_dout[51]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[51]),
      .w_in_read(pe_w_out_w_in_read[51]),
      .w_out_din(pe_w_out_w_in_din[52]),
      .w_out_full_n(pe_w_out_w_in_full_n[52]),
      .w_out_write(pe_w_out_w_in_write[52]));
  pe_r0 u_pe_r0_6_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[52]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[52]),
      .a_in_read(pe_a_out_a_in_read[52]),
      .a_out_din(pe_a_out_a_in_din[53]),
      .a_out_full_n(pe_a_out_a_in_full_n[53]),
      .a_out_write(pe_a_out_a_in_write[53]),
      .p_in_dout(pe_p_out_p_in_dout[52]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[52]),
      .p_in_read(pe_p_out_p_in_read[52]),
      .p_out_din(pe_p_out_p_in_din[60]),
      .p_out_full_n(pe_p_out_p_in_full_n[60]),
      .p_out_write(pe_p_out_p_in_write[60]),
      .w_in_dout(pe_w_out_w_in_dout[52]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[52]),
      .w_in_read(pe_w_out_w_in_read[52]),
      .w_out_din(pe_w_out_w_in_din[53]),
      .w_out_full_n(pe_w_out_w_in_full_n[53]),
      .w_out_write(pe_w_out_w_in_write[53]));
  pe_r0 u_pe_r0_6_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[53]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[53]),
      .a_in_read(pe_a_out_a_in_read[53]),
      .a_out_din(pe_a_out_a_in_din[54]),
      .a_out_full_n(pe_a_out_a_in_full_n[54]),
      .a_out_write(pe_a_out_a_in_write[54]),
      .p_in_dout(pe_p_out_p_in_dout[53]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[53]),
      .p_in_read(pe_p_out_p_in_read[53]),
      .p_out_din(pe_p_out_p_in_din[61]),
      .p_out_full_n(pe_p_out_p_in_full_n[61]),
      .p_out_write(pe_p_out_p_in_write[61]),
      .w_in_dout(pe_w_out_w_in_dout[53]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[53]),
      .w_in_read(pe_w_out_w_in_read[53]),
      .w_out_din(pe_w_out_w_in_din[54]),
      .w_out_full_n(pe_w_out_w_in_full_n[54]),
      .w_out_write(pe_w_out_w_in_write[54]));
  pe_r0 u_pe_r0_6_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[54]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[54]),
      .a_in_read(pe_a_out_a_in_read[54]),
      .a_out_din(pe_a_out_a_in_din[55]),
      .a_out_full_n(pe_a_out_a_in_full_n[55]),
      .a_out_write(pe_a_out_a_in_write[55]),
      .p_in_dout(pe_p_out_p_in_dout[54]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[54]),
      .p_in_read(pe_p_out_p_in_read[54]),
      .p_out_din(pe_p_out_p_in_din[62]),
      .p_out_full_n(pe_p_out_p_in_full_n[62]),
      .p_out_write(pe_p_out_p_in_write[62]),
      .w_in_dout(pe_w_out_w_in_dout[54]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[54]),
      .w_in_read(pe_w_out_w_in_read[54]),
      .w_out_din(pe_w_out_w_in_din[55]),
      .w_out_full_n(pe_w_out_w_in_full_n[55]),
      .w_out_write(pe_w_out_w_in_write[55]));
  // role pe_r1: 6 instance(s)
  pe_r1 u_pe_r1_7_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[57]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[57]),
      .a_in_read(pe_a_out_a_in_read[57]),
      .a_out_din(pe_a_out_a_in_din[58]),
      .a_out_full_n(pe_a_out_a_in_full_n[58]),
      .a_out_write(pe_a_out_a_in_write[58]),
      .p_in_dout(pe_p_out_p_in_dout[57]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[57]),
      .p_in_read(pe_p_out_p_in_read[57]),
      .p_out_din(lane_z_in_bind_din[1]),
      .p_out_full_n(lane_z_in_bind_full_n[1]),
      .p_out_write(lane_z_in_bind_write[1]),
      .w_in_dout(pe_w_out_w_in_dout[57]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[57]),
      .w_in_read(pe_w_out_w_in_read[57]),
      .w_out_din(pe_w_out_w_in_din[58]),
      .w_out_full_n(pe_w_out_w_in_full_n[58]),
      .w_out_write(pe_w_out_w_in_write[58]));
  pe_r1 u_pe_r1_7_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[58]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[58]),
      .a_in_read(pe_a_out_a_in_read[58]),
      .a_out_din(pe_a_out_a_in_din[59]),
      .a_out_full_n(pe_a_out_a_in_full_n[59]),
      .a_out_write(pe_a_out_a_in_write[59]),
      .p_in_dout(pe_p_out_p_in_dout[58]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[58]),
      .p_in_read(pe_p_out_p_in_read[58]),
      .p_out_din(lane_z_in_bind_din[2]),
      .p_out_full_n(lane_z_in_bind_full_n[2]),
      .p_out_write(lane_z_in_bind_write[2]),
      .w_in_dout(pe_w_out_w_in_dout[58]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[58]),
      .w_in_read(pe_w_out_w_in_read[58]),
      .w_out_din(pe_w_out_w_in_din[59]),
      .w_out_full_n(pe_w_out_w_in_full_n[59]),
      .w_out_write(pe_w_out_w_in_write[59]));
  pe_r1 u_pe_r1_7_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[59]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[59]),
      .a_in_read(pe_a_out_a_in_read[59]),
      .a_out_din(pe_a_out_a_in_din[60]),
      .a_out_full_n(pe_a_out_a_in_full_n[60]),
      .a_out_write(pe_a_out_a_in_write[60]),
      .p_in_dout(pe_p_out_p_in_dout[59]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[59]),
      .p_in_read(pe_p_out_p_in_read[59]),
      .p_out_din(lane_z_in_bind_din[3]),
      .p_out_full_n(lane_z_in_bind_full_n[3]),
      .p_out_write(lane_z_in_bind_write[3]),
      .w_in_dout(pe_w_out_w_in_dout[59]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[59]),
      .w_in_read(pe_w_out_w_in_read[59]),
      .w_out_din(pe_w_out_w_in_din[60]),
      .w_out_full_n(pe_w_out_w_in_full_n[60]),
      .w_out_write(pe_w_out_w_in_write[60]));
  pe_r1 u_pe_r1_7_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[60]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[60]),
      .a_in_read(pe_a_out_a_in_read[60]),
      .a_out_din(pe_a_out_a_in_din[61]),
      .a_out_full_n(pe_a_out_a_in_full_n[61]),
      .a_out_write(pe_a_out_a_in_write[61]),
      .p_in_dout(pe_p_out_p_in_dout[60]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[60]),
      .p_in_read(pe_p_out_p_in_read[60]),
      .p_out_din(lane_z_in_bind_din[4]),
      .p_out_full_n(lane_z_in_bind_full_n[4]),
      .p_out_write(lane_z_in_bind_write[4]),
      .w_in_dout(pe_w_out_w_in_dout[60]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[60]),
      .w_in_read(pe_w_out_w_in_read[60]),
      .w_out_din(pe_w_out_w_in_din[61]),
      .w_out_full_n(pe_w_out_w_in_full_n[61]),
      .w_out_write(pe_w_out_w_in_write[61]));
  pe_r1 u_pe_r1_7_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[61]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[61]),
      .a_in_read(pe_a_out_a_in_read[61]),
      .a_out_din(pe_a_out_a_in_din[62]),
      .a_out_full_n(pe_a_out_a_in_full_n[62]),
      .a_out_write(pe_a_out_a_in_write[62]),
      .p_in_dout(pe_p_out_p_in_dout[61]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[61]),
      .p_in_read(pe_p_out_p_in_read[61]),
      .p_out_din(lane_z_in_bind_din[5]),
      .p_out_full_n(lane_z_in_bind_full_n[5]),
      .p_out_write(lane_z_in_bind_write[5]),
      .w_in_dout(pe_w_out_w_in_dout[61]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[61]),
      .w_in_read(pe_w_out_w_in_read[61]),
      .w_out_din(pe_w_out_w_in_din[62]),
      .w_out_full_n(pe_w_out_w_in_full_n[62]),
      .w_out_write(pe_w_out_w_in_write[62]));
  pe_r1 u_pe_r1_7_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[62]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[62]),
      .a_in_read(pe_a_out_a_in_read[62]),
      .a_out_din(pe_a_out_a_in_din[63]),
      .a_out_full_n(pe_a_out_a_in_full_n[63]),
      .a_out_write(pe_a_out_a_in_write[63]),
      .p_in_dout(pe_p_out_p_in_dout[62]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[62]),
      .p_in_read(pe_p_out_p_in_read[62]),
      .p_out_din(lane_z_in_bind_din[6]),
      .p_out_full_n(lane_z_in_bind_full_n[6]),
      .p_out_write(lane_z_in_bind_write[6]),
      .w_in_dout(pe_w_out_w_in_dout[62]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[62]),
      .w_in_read(pe_w_out_w_in_read[62]),
      .w_out_din(pe_w_out_w_in_din[63]),
      .w_out_full_n(pe_w_out_w_in_full_n[63]),
      .w_out_write(pe_w_out_w_in_write[63]));
  // role pe_r2: 6 instance(s)
  pe_r2 u_pe_r2_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[1]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[1]),
      .a_in_read(pe_a_out_a_in_read[1]),
      .a_out_din(pe_a_out_a_in_din[2]),
      .a_out_full_n(pe_a_out_a_in_full_n[2]),
      .a_out_write(pe_a_out_a_in_write[2]),
      .p_out_din(pe_p_out_p_in_din[9]),
      .p_out_full_n(pe_p_out_p_in_full_n[9]),
      .p_out_write(pe_p_out_p_in_write[9]),
      .w_in_dout(pe_w_out_w_in_dout[1]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[1]),
      .w_in_read(pe_w_out_w_in_read[1]),
      .w_out_din(pe_w_out_w_in_din[2]),
      .w_out_full_n(pe_w_out_w_in_full_n[2]),
      .w_out_write(pe_w_out_w_in_write[2]));
  pe_r2 u_pe_r2_0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[2]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[2]),
      .a_in_read(pe_a_out_a_in_read[2]),
      .a_out_din(pe_a_out_a_in_din[3]),
      .a_out_full_n(pe_a_out_a_in_full_n[3]),
      .a_out_write(pe_a_out_a_in_write[3]),
      .p_out_din(pe_p_out_p_in_din[10]),
      .p_out_full_n(pe_p_out_p_in_full_n[10]),
      .p_out_write(pe_p_out_p_in_write[10]),
      .w_in_dout(pe_w_out_w_in_dout[2]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[2]),
      .w_in_read(pe_w_out_w_in_read[2]),
      .w_out_din(pe_w_out_w_in_din[3]),
      .w_out_full_n(pe_w_out_w_in_full_n[3]),
      .w_out_write(pe_w_out_w_in_write[3]));
  pe_r2 u_pe_r2_0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[3]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[3]),
      .a_in_read(pe_a_out_a_in_read[3]),
      .a_out_din(pe_a_out_a_in_din[4]),
      .a_out_full_n(pe_a_out_a_in_full_n[4]),
      .a_out_write(pe_a_out_a_in_write[4]),
      .p_out_din(pe_p_out_p_in_din[11]),
      .p_out_full_n(pe_p_out_p_in_full_n[11]),
      .p_out_write(pe_p_out_p_in_write[11]),
      .w_in_dout(pe_w_out_w_in_dout[3]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[3]),
      .w_in_read(pe_w_out_w_in_read[3]),
      .w_out_din(pe_w_out_w_in_din[4]),
      .w_out_full_n(pe_w_out_w_in_full_n[4]),
      .w_out_write(pe_w_out_w_in_write[4]));
  pe_r2 u_pe_r2_0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[4]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[4]),
      .a_in_read(pe_a_out_a_in_read[4]),
      .a_out_din(pe_a_out_a_in_din[5]),
      .a_out_full_n(pe_a_out_a_in_full_n[5]),
      .a_out_write(pe_a_out_a_in_write[5]),
      .p_out_din(pe_p_out_p_in_din[12]),
      .p_out_full_n(pe_p_out_p_in_full_n[12]),
      .p_out_write(pe_p_out_p_in_write[12]),
      .w_in_dout(pe_w_out_w_in_dout[4]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[4]),
      .w_in_read(pe_w_out_w_in_read[4]),
      .w_out_din(pe_w_out_w_in_din[5]),
      .w_out_full_n(pe_w_out_w_in_full_n[5]),
      .w_out_write(pe_w_out_w_in_write[5]));
  pe_r2 u_pe_r2_0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[5]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[5]),
      .a_in_read(pe_a_out_a_in_read[5]),
      .a_out_din(pe_a_out_a_in_din[6]),
      .a_out_full_n(pe_a_out_a_in_full_n[6]),
      .a_out_write(pe_a_out_a_in_write[6]),
      .p_out_din(pe_p_out_p_in_din[13]),
      .p_out_full_n(pe_p_out_p_in_full_n[13]),
      .p_out_write(pe_p_out_p_in_write[13]),
      .w_in_dout(pe_w_out_w_in_dout[5]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[5]),
      .w_in_read(pe_w_out_w_in_read[5]),
      .w_out_din(pe_w_out_w_in_din[6]),
      .w_out_full_n(pe_w_out_w_in_full_n[6]),
      .w_out_write(pe_w_out_w_in_write[6]));
  pe_r2 u_pe_r2_0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[6]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[6]),
      .a_in_read(pe_a_out_a_in_read[6]),
      .a_out_din(pe_a_out_a_in_din[7]),
      .a_out_full_n(pe_a_out_a_in_full_n[7]),
      .a_out_write(pe_a_out_a_in_write[7]),
      .p_out_din(pe_p_out_p_in_din[14]),
      .p_out_full_n(pe_p_out_p_in_full_n[14]),
      .p_out_write(pe_p_out_p_in_write[14]),
      .w_in_dout(pe_w_out_w_in_dout[6]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[6]),
      .w_in_read(pe_w_out_w_in_read[6]),
      .w_out_din(pe_w_out_w_in_din[7]),
      .w_out_full_n(pe_w_out_w_in_full_n[7]),
      .w_out_write(pe_w_out_w_in_write[7]));
  // role pe_r3: 6 instance(s)
  pe_r3 u_pe_r3_1_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[15]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[15]),
      .a_in_read(pe_a_out_a_in_read[15]),
      .p_in_dout(pe_p_out_p_in_dout[15]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[15]),
      .p_in_read(pe_p_out_p_in_read[15]),
      .p_out_din(pe_p_out_p_in_din[23]),
      .p_out_full_n(pe_p_out_p_in_full_n[23]),
      .p_out_write(pe_p_out_p_in_write[23]),
      .w_in_dout(pe_w_out_w_in_dout[15]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[15]),
      .w_in_read(pe_w_out_w_in_read[15]));
  pe_r3 u_pe_r3_2_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[23]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[23]),
      .a_in_read(pe_a_out_a_in_read[23]),
      .p_in_dout(pe_p_out_p_in_dout[23]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[23]),
      .p_in_read(pe_p_out_p_in_read[23]),
      .p_out_din(pe_p_out_p_in_din[31]),
      .p_out_full_n(pe_p_out_p_in_full_n[31]),
      .p_out_write(pe_p_out_p_in_write[31]),
      .w_in_dout(pe_w_out_w_in_dout[23]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[23]),
      .w_in_read(pe_w_out_w_in_read[23]));
  pe_r3 u_pe_r3_3_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[31]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[31]),
      .a_in_read(pe_a_out_a_in_read[31]),
      .p_in_dout(pe_p_out_p_in_dout[31]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[31]),
      .p_in_read(pe_p_out_p_in_read[31]),
      .p_out_din(pe_p_out_p_in_din[39]),
      .p_out_full_n(pe_p_out_p_in_full_n[39]),
      .p_out_write(pe_p_out_p_in_write[39]),
      .w_in_dout(pe_w_out_w_in_dout[31]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[31]),
      .w_in_read(pe_w_out_w_in_read[31]));
  pe_r3 u_pe_r3_4_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[39]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[39]),
      .a_in_read(pe_a_out_a_in_read[39]),
      .p_in_dout(pe_p_out_p_in_dout[39]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[39]),
      .p_in_read(pe_p_out_p_in_read[39]),
      .p_out_din(pe_p_out_p_in_din[47]),
      .p_out_full_n(pe_p_out_p_in_full_n[47]),
      .p_out_write(pe_p_out_p_in_write[47]),
      .w_in_dout(pe_w_out_w_in_dout[39]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[39]),
      .w_in_read(pe_w_out_w_in_read[39]));
  pe_r3 u_pe_r3_5_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[47]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[47]),
      .a_in_read(pe_a_out_a_in_read[47]),
      .p_in_dout(pe_p_out_p_in_dout[47]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[47]),
      .p_in_read(pe_p_out_p_in_read[47]),
      .p_out_din(pe_p_out_p_in_din[55]),
      .p_out_full_n(pe_p_out_p_in_full_n[55]),
      .p_out_write(pe_p_out_p_in_write[55]),
      .w_in_dout(pe_w_out_w_in_dout[47]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[47]),
      .w_in_read(pe_w_out_w_in_read[47]));
  pe_r3 u_pe_r3_6_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[55]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[55]),
      .a_in_read(pe_a_out_a_in_read[55]),
      .p_in_dout(pe_p_out_p_in_dout[55]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[55]),
      .p_in_read(pe_p_out_p_in_read[55]),
      .p_out_din(pe_p_out_p_in_din[63]),
      .p_out_full_n(pe_p_out_p_in_full_n[63]),
      .p_out_write(pe_p_out_p_in_write[63]),
      .w_in_dout(pe_w_out_w_in_dout[55]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[55]),
      .w_in_read(pe_w_out_w_in_read[55]));
  // role pe_r4: 6 instance(s)
  pe_r4 u_pe_r4_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_in_bind_dout[1]),
      .a_in_empty_n(pe_a_in_bind_empty_n[1]),
      .a_in_read(pe_a_in_bind_read[1]),
      .a_out_din(pe_a_out_a_in_din[9]),
      .a_out_full_n(pe_a_out_a_in_full_n[9]),
      .a_out_write(pe_a_out_a_in_write[9]),
      .p_in_dout(pe_p_out_p_in_dout[8]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[8]),
      .p_in_read(pe_p_out_p_in_read[8]),
      .p_out_din(pe_p_out_p_in_din[16]),
      .p_out_full_n(pe_p_out_p_in_full_n[16]),
      .p_out_write(pe_p_out_p_in_write[16]),
      .w_in_dout(pe_w_in_bind_dout[1]),
      .w_in_empty_n(pe_w_in_bind_empty_n[1]),
      .w_in_read(pe_w_in_bind_read[1]),
      .w_out_din(pe_w_out_w_in_din[9]),
      .w_out_full_n(pe_w_out_w_in_full_n[9]),
      .w_out_write(pe_w_out_w_in_write[9]));
  pe_r4 u_pe_r4_2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_in_bind_dout[2]),
      .a_in_empty_n(pe_a_in_bind_empty_n[2]),
      .a_in_read(pe_a_in_bind_read[2]),
      .a_out_din(pe_a_out_a_in_din[17]),
      .a_out_full_n(pe_a_out_a_in_full_n[17]),
      .a_out_write(pe_a_out_a_in_write[17]),
      .p_in_dout(pe_p_out_p_in_dout[16]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[16]),
      .p_in_read(pe_p_out_p_in_read[16]),
      .p_out_din(pe_p_out_p_in_din[24]),
      .p_out_full_n(pe_p_out_p_in_full_n[24]),
      .p_out_write(pe_p_out_p_in_write[24]),
      .w_in_dout(pe_w_in_bind_dout[2]),
      .w_in_empty_n(pe_w_in_bind_empty_n[2]),
      .w_in_read(pe_w_in_bind_read[2]),
      .w_out_din(pe_w_out_w_in_din[17]),
      .w_out_full_n(pe_w_out_w_in_full_n[17]),
      .w_out_write(pe_w_out_w_in_write[17]));
  pe_r4 u_pe_r4_3_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_in_bind_dout[3]),
      .a_in_empty_n(pe_a_in_bind_empty_n[3]),
      .a_in_read(pe_a_in_bind_read[3]),
      .a_out_din(pe_a_out_a_in_din[25]),
      .a_out_full_n(pe_a_out_a_in_full_n[25]),
      .a_out_write(pe_a_out_a_in_write[25]),
      .p_in_dout(pe_p_out_p_in_dout[24]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[24]),
      .p_in_read(pe_p_out_p_in_read[24]),
      .p_out_din(pe_p_out_p_in_din[32]),
      .p_out_full_n(pe_p_out_p_in_full_n[32]),
      .p_out_write(pe_p_out_p_in_write[32]),
      .w_in_dout(pe_w_in_bind_dout[3]),
      .w_in_empty_n(pe_w_in_bind_empty_n[3]),
      .w_in_read(pe_w_in_bind_read[3]),
      .w_out_din(pe_w_out_w_in_din[25]),
      .w_out_full_n(pe_w_out_w_in_full_n[25]),
      .w_out_write(pe_w_out_w_in_write[25]));
  pe_r4 u_pe_r4_4_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_in_bind_dout[4]),
      .a_in_empty_n(pe_a_in_bind_empty_n[4]),
      .a_in_read(pe_a_in_bind_read[4]),
      .a_out_din(pe_a_out_a_in_din[33]),
      .a_out_full_n(pe_a_out_a_in_full_n[33]),
      .a_out_write(pe_a_out_a_in_write[33]),
      .p_in_dout(pe_p_out_p_in_dout[32]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[32]),
      .p_in_read(pe_p_out_p_in_read[32]),
      .p_out_din(pe_p_out_p_in_din[40]),
      .p_out_full_n(pe_p_out_p_in_full_n[40]),
      .p_out_write(pe_p_out_p_in_write[40]),
      .w_in_dout(pe_w_in_bind_dout[4]),
      .w_in_empty_n(pe_w_in_bind_empty_n[4]),
      .w_in_read(pe_w_in_bind_read[4]),
      .w_out_din(pe_w_out_w_in_din[33]),
      .w_out_full_n(pe_w_out_w_in_full_n[33]),
      .w_out_write(pe_w_out_w_in_write[33]));
  pe_r4 u_pe_r4_5_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_in_bind_dout[5]),
      .a_in_empty_n(pe_a_in_bind_empty_n[5]),
      .a_in_read(pe_a_in_bind_read[5]),
      .a_out_din(pe_a_out_a_in_din[41]),
      .a_out_full_n(pe_a_out_a_in_full_n[41]),
      .a_out_write(pe_a_out_a_in_write[41]),
      .p_in_dout(pe_p_out_p_in_dout[40]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[40]),
      .p_in_read(pe_p_out_p_in_read[40]),
      .p_out_din(pe_p_out_p_in_din[48]),
      .p_out_full_n(pe_p_out_p_in_full_n[48]),
      .p_out_write(pe_p_out_p_in_write[48]),
      .w_in_dout(pe_w_in_bind_dout[5]),
      .w_in_empty_n(pe_w_in_bind_empty_n[5]),
      .w_in_read(pe_w_in_bind_read[5]),
      .w_out_din(pe_w_out_w_in_din[41]),
      .w_out_full_n(pe_w_out_w_in_full_n[41]),
      .w_out_write(pe_w_out_w_in_write[41]));
  pe_r4 u_pe_r4_6_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_in_bind_dout[6]),
      .a_in_empty_n(pe_a_in_bind_empty_n[6]),
      .a_in_read(pe_a_in_bind_read[6]),
      .a_out_din(pe_a_out_a_in_din[49]),
      .a_out_full_n(pe_a_out_a_in_full_n[49]),
      .a_out_write(pe_a_out_a_in_write[49]),
      .p_in_dout(pe_p_out_p_in_dout[48]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[48]),
      .p_in_read(pe_p_out_p_in_read[48]),
      .p_out_din(pe_p_out_p_in_din[56]),
      .p_out_full_n(pe_p_out_p_in_full_n[56]),
      .p_out_write(pe_p_out_p_in_write[56]),
      .w_in_dout(pe_w_in_bind_dout[6]),
      .w_in_empty_n(pe_w_in_bind_empty_n[6]),
      .w_in_read(pe_w_in_bind_read[6]),
      .w_out_din(pe_w_out_w_in_din[49]),
      .w_out_full_n(pe_w_out_w_in_full_n[49]),
      .w_out_write(pe_w_out_w_in_write[49]));
  // role pe_r5: 1 instance(s)
  pe_r5 u_pe_r5_7_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[63]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[63]),
      .a_in_read(pe_a_out_a_in_read[63]),
      .p_in_dout(pe_p_out_p_in_dout[63]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[63]),
      .p_in_read(pe_p_out_p_in_read[63]),
      .p_out_din(lane_z_in_bind_din[7]),
      .p_out_full_n(lane_z_in_bind_full_n[7]),
      .p_out_write(lane_z_in_bind_write[7]),
      .w_in_dout(pe_w_out_w_in_dout[63]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[63]),
      .w_in_read(pe_w_out_w_in_read[63]));
  // role pe_r6: 1 instance(s)
  pe_r6 u_pe_r6_0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[7]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[7]),
      .a_in_read(pe_a_out_a_in_read[7]),
      .p_out_din(pe_p_out_p_in_din[15]),
      .p_out_full_n(pe_p_out_p_in_full_n[15]),
      .p_out_write(pe_p_out_p_in_write[15]),
      .w_in_dout(pe_w_out_w_in_dout[7]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[7]),
      .w_in_read(pe_w_out_w_in_read[7]));
  // role pe_r7: 1 instance(s)
  pe_r7 u_pe_r7_7_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_in_bind_dout[7]),
      .a_in_empty_n(pe_a_in_bind_empty_n[7]),
      .a_in_read(pe_a_in_bind_read[7]),
      .a_out_din(pe_a_out_a_in_din[57]),
      .a_out_full_n(pe_a_out_a_in_full_n[57]),
      .a_out_write(pe_a_out_a_in_write[57]),
      .p_in_dout(pe_p_out_p_in_dout[56]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[56]),
      .p_in_read(pe_p_out_p_in_read[56]),
      .p_out_din(lane_z_in_bind_din[0]),
      .p_out_full_n(lane_z_in_bind_full_n[0]),
      .p_out_write(lane_z_in_bind_write[0]),
      .w_in_dout(pe_w_in_bind_dout[7]),
      .w_in_empty_n(pe_w_in_bind_empty_n[7]),
      .w_in_read(pe_w_in_bind_read[7]),
      .w_out_din(pe_w_out_w_in_din[57]),
      .w_out_full_n(pe_w_out_w_in_full_n[57]),
      .w_out_write(pe_w_out_w_in_write[57]));
  // role pe_r8: 1 instance(s)
  pe_r8 u_pe_r8_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_in_bind_dout[0]),
      .a_in_empty_n(pe_a_in_bind_empty_n[0]),
      .a_in_read(pe_a_in_bind_read[0]),
      .a_out_din(pe_a_out_a_in_din[1]),
      .a_out_full_n(pe_a_out_a_in_full_n[1]),
      .a_out_write(pe_a_out_a_in_write[1]),
      .p_out_din(pe_p_out_p_in_din[8]),
      .p_out_full_n(pe_p_out_p_in_full_n[8]),
      .p_out_write(pe_p_out_p_in_write[8]),
      .w_in_dout(pe_w_in_bind_dout[0]),
      .w_in_empty_n(pe_w_in_bind_empty_n[0]),
      .w_in_read(pe_w_in_bind_read[0]),
      .w_out_din(pe_w_out_w_in_din[1]),
      .w_out_full_n(pe_w_out_w_in_full_n[1]),
      .w_out_write(pe_w_out_w_in_write[1]));
  // role etap_r0: 6 instance(s)
  etap_r0 u_etap_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_out_din(pe_a_in_bind_din[1]),
      .a_out_full_n(pe_a_in_bind_full_n[1]),
      .a_out_write(pe_a_in_bind_write[1]),
      .e_in_dout(etap_e_out_e_in_dout[1]),
      .e_in_empty_n(etap_e_out_e_in_empty_n[1]),
      .e_in_read(etap_e_out_e_in_read[1]),
      .e_out_din(etap_e_out_e_in_din[2]),
      .e_out_full_n(etap_e_out_e_in_full_n[2]),
      .e_out_write(etap_e_out_e_in_write[2]),
      .w_out_din(pe_w_in_bind_din[1]),
      .w_out_full_n(pe_w_in_bind_full_n[1]),
      .w_out_write(pe_w_in_bind_write[1]));
  etap_r0 u_etap_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_out_din(pe_a_in_bind_din[2]),
      .a_out_full_n(pe_a_in_bind_full_n[2]),
      .a_out_write(pe_a_in_bind_write[2]),
      .e_in_dout(etap_e_out_e_in_dout[2]),
      .e_in_empty_n(etap_e_out_e_in_empty_n[2]),
      .e_in_read(etap_e_out_e_in_read[2]),
      .e_out_din(etap_e_out_e_in_din[3]),
      .e_out_full_n(etap_e_out_e_in_full_n[3]),
      .e_out_write(etap_e_out_e_in_write[3]),
      .w_out_din(pe_w_in_bind_din[2]),
      .w_out_full_n(pe_w_in_bind_full_n[2]),
      .w_out_write(pe_w_in_bind_write[2]));
  etap_r0 u_etap_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_out_din(pe_a_in_bind_din[3]),
      .a_out_full_n(pe_a_in_bind_full_n[3]),
      .a_out_write(pe_a_in_bind_write[3]),
      .e_in_dout(etap_e_out_e_in_dout[3]),
      .e_in_empty_n(etap_e_out_e_in_empty_n[3]),
      .e_in_read(etap_e_out_e_in_read[3]),
      .e_out_din(etap_e_out_e_in_din[4]),
      .e_out_full_n(etap_e_out_e_in_full_n[4]),
      .e_out_write(etap_e_out_e_in_write[4]),
      .w_out_din(pe_w_in_bind_din[3]),
      .w_out_full_n(pe_w_in_bind_full_n[3]),
      .w_out_write(pe_w_in_bind_write[3]));
  etap_r0 u_etap_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_out_din(pe_a_in_bind_din[4]),
      .a_out_full_n(pe_a_in_bind_full_n[4]),
      .a_out_write(pe_a_in_bind_write[4]),
      .e_in_dout(etap_e_out_e_in_dout[4]),
      .e_in_empty_n(etap_e_out_e_in_empty_n[4]),
      .e_in_read(etap_e_out_e_in_read[4]),
      .e_out_din(etap_e_out_e_in_din[5]),
      .e_out_full_n(etap_e_out_e_in_full_n[5]),
      .e_out_write(etap_e_out_e_in_write[5]),
      .w_out_din(pe_w_in_bind_din[4]),
      .w_out_full_n(pe_w_in_bind_full_n[4]),
      .w_out_write(pe_w_in_bind_write[4]));
  etap_r0 u_etap_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_out_din(pe_a_in_bind_din[5]),
      .a_out_full_n(pe_a_in_bind_full_n[5]),
      .a_out_write(pe_a_in_bind_write[5]),
      .e_in_dout(etap_e_out_e_in_dout[5]),
      .e_in_empty_n(etap_e_out_e_in_empty_n[5]),
      .e_in_read(etap_e_out_e_in_read[5]),
      .e_out_din(etap_e_out_e_in_din[6]),
      .e_out_full_n(etap_e_out_e_in_full_n[6]),
      .e_out_write(etap_e_out_e_in_write[6]),
      .w_out_din(pe_w_in_bind_din[5]),
      .w_out_full_n(pe_w_in_bind_full_n[5]),
      .w_out_write(pe_w_in_bind_write[5]));
  etap_r0 u_etap_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_out_din(pe_a_in_bind_din[6]),
      .a_out_full_n(pe_a_in_bind_full_n[6]),
      .a_out_write(pe_a_in_bind_write[6]),
      .e_in_dout(etap_e_out_e_in_dout[6]),
      .e_in_empty_n(etap_e_out_e_in_empty_n[6]),
      .e_in_read(etap_e_out_e_in_read[6]),
      .e_out_din(etap_e_out_e_in_din[7]),
      .e_out_full_n(etap_e_out_e_in_full_n[7]),
      .e_out_write(etap_e_out_e_in_write[7]),
      .w_out_din(pe_w_in_bind_din[6]),
      .w_out_full_n(pe_w_in_bind_full_n[6]),
      .w_out_write(pe_w_in_bind_write[6]));
  // role etap_r1: 1 instance(s)
  etap_r1 u_etap_r1_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_out_din(pe_a_in_bind_din[7]),
      .a_out_full_n(pe_a_in_bind_full_n[7]),
      .a_out_write(pe_a_in_bind_write[7]),
      .e_in_dout(etap_e_out_e_in_dout[7]),
      .e_in_empty_n(etap_e_out_e_in_empty_n[7]),
      .e_in_read(etap_e_out_e_in_read[7]),
      .w_out_din(pe_w_in_bind_din[7]),
      .w_out_full_n(pe_w_in_bind_full_n[7]),
      .w_out_write(pe_w_in_bind_write[7]));
  // role etap_r2: 1 instance(s)
  etap_r2 u_etap_r2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_out_din(pe_a_in_bind_din[0]),
      .a_out_full_n(pe_a_in_bind_full_n[0]),
      .a_out_write(pe_a_in_bind_write[0]),
      .e_in_dout(etap_e_in_bind_dout[0]),
      .e_in_empty_n(etap_e_in_bind_empty_n[0]),
      .e_in_read(etap_e_in_bind_read[0]),
      .e_out_din(etap_e_out_e_in_din[1]),
      .e_out_full_n(etap_e_out_e_in_full_n[1]),
      .e_out_write(etap_e_out_e_in_write[1]),
      .w_out_din(pe_w_in_bind_din[0]),
      .w_out_full_n(pe_w_in_bind_full_n[0]),
      .w_out_write(pe_w_in_bind_write[0]));
  // role req8_r0: 1 instance(s)
  req8_r0 u_req8_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .ins_in_dout(req8_ins_in_bind_dout[0]),
      .ins_in_empty_n(req8_ins_in_bind_empty_n[0]),
      .ins_in_read(req8_ins_in_bind_read[0]),
      .launch_dout(req8_launch_bind_dout[0]),
      .launch_empty_n(req8_launch_bind_empty_n[0]),
      .launch_read(req8_launch_bind_read[0]),
      .op_out_din(head8_op_in_bind_din[0]),
      .op_out_full_n(head8_op_in_bind_full_n[0]),
      .op_out_write(head8_op_in_bind_write[0]),
      .rd_cmd_din(req8_rd_cmd_bind_din[0]),
      .rd_cmd_full_n(req8_rd_cmd_bind_full_n[0]),
      .rd_cmd_write(req8_rd_cmd_bind_write[0]),
      .tag_out_din(deal8_tag_in_bind_din[0]),
      .tag_out_full_n(deal8_tag_in_bind_full_n[0]),
      .tag_out_write(deal8_tag_in_bind_write[0]),
      .y_out_din(wreq8_y_cmd_bind_din[0]),
      .y_out_full_n(wreq8_y_cmd_bind_full_n[0]),
      .y_out_write(wreq8_y_cmd_bind_write[0]));
  // role deal8_r0: 1 instance(s)
  deal8_r0 u_deal8_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_out_din(head8_a_in_bind_din[0]),
      .a_out_full_n(head8_a_in_bind_full_n[0]),
      .a_out_write(head8_a_in_bind_write[0]),
      .b_out_din(head8_b_in_bind_din[0]),
      .b_out_full_n(head8_b_in_bind_full_n[0]),
      .b_out_write(head8_b_in_bind_write[0]),
      .ins_out_din(req8_ins_in_bind_din[0]),
      .ins_out_full_n(req8_ins_in_bind_full_n[0]),
      .ins_out_write(req8_ins_in_bind_write[0]),
      .rd_data_dout(deal8_rd_data_bind_dout[0]),
      .rd_data_empty_n(deal8_rd_data_bind_empty_n[0]),
      .rd_data_read(deal8_rd_data_bind_read[0]),
      .tag_in_dout(deal8_tag_in_bind_dout[0]),
      .tag_in_empty_n(deal8_tag_in_bind_empty_n[0]),
      .tag_in_read(deal8_tag_in_bind_read[0]),
      .w_out_din(head8_w_in_bind_din[0]),
      .w_out_full_n(head8_w_in_bind_full_n[0]),
      .w_out_write(head8_w_in_bind_write[0]));
  // role head8_r0: 1 instance(s)
  head8_r0 u_head8_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(head8_a_in_bind_dout[0]),
      .a_in_empty_n(head8_a_in_bind_empty_n[0]),
      .a_in_read(head8_a_in_bind_read[0]),
      .b_in_dout(head8_b_in_bind_dout[0]),
      .b_in_empty_n(head8_b_in_bind_empty_n[0]),
      .b_in_read(head8_b_in_bind_read[0]),
      .credit_dout(head8_credit_bind_dout[0]),
      .credit_empty_n(head8_credit_bind_empty_n[0]),
      .credit_read(head8_credit_bind_read[0]),
      .e_out_din(etap_e_in_bind_din[0]),
      .e_out_full_n(etap_e_in_bind_full_n[0]),
      .e_out_write(etap_e_in_bind_write[0]),
      .op_in_dout(head8_op_in_bind_dout[0]),
      .op_in_empty_n(head8_op_in_bind_empty_n[0]),
      .op_in_read(head8_op_in_bind_read[0]),
      .u_out_din(uq_u_in_bind_din[0]),
      .u_out_full_n(uq_u_in_bind_full_n[0]),
      .u_out_write(uq_u_in_bind_write[0]),
      .w_in_dout(head8_w_in_bind_dout[0]),
      .w_in_empty_n(head8_w_in_bind_empty_n[0]),
      .w_in_read(head8_w_in_bind_read[0]));
  // role uq_r0: 1 instance(s)
  uq_r0 u_uq_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(uq_u_in_bind_dout[0]),
      .u_in_empty_n(uq_u_in_bind_empty_n[0]),
      .u_in_read(uq_u_in_bind_read[0]),
      .u_out_din(tap_u_in_bind_din[0]),
      .u_out_full_n(tap_u_in_bind_full_n[0]),
      .u_out_write(tap_u_in_bind_write[0]));
  // role tap_r0: 6 instance(s)
  tap_r0 u_tap_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_out_din(lane_c_in_bind_din[1]),
      .c_out_full_n(lane_c_in_bind_full_n[1]),
      .c_out_write(lane_c_in_bind_write[1]),
      .u_in_dout(tap_u_out_u_in_dout[1]),
      .u_in_empty_n(tap_u_out_u_in_empty_n[1]),
      .u_in_read(tap_u_out_u_in_read[1]),
      .u_out_din(tap_u_out_u_in_din[2]),
      .u_out_full_n(tap_u_out_u_in_full_n[2]),
      .u_out_write(tap_u_out_u_in_write[2]));
  tap_r0 u_tap_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_out_din(lane_c_in_bind_din[2]),
      .c_out_full_n(lane_c_in_bind_full_n[2]),
      .c_out_write(lane_c_in_bind_write[2]),
      .u_in_dout(tap_u_out_u_in_dout[2]),
      .u_in_empty_n(tap_u_out_u_in_empty_n[2]),
      .u_in_read(tap_u_out_u_in_read[2]),
      .u_out_din(tap_u_out_u_in_din[3]),
      .u_out_full_n(tap_u_out_u_in_full_n[3]),
      .u_out_write(tap_u_out_u_in_write[3]));
  tap_r0 u_tap_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_out_din(lane_c_in_bind_din[3]),
      .c_out_full_n(lane_c_in_bind_full_n[3]),
      .c_out_write(lane_c_in_bind_write[3]),
      .u_in_dout(tap_u_out_u_in_dout[3]),
      .u_in_empty_n(tap_u_out_u_in_empty_n[3]),
      .u_in_read(tap_u_out_u_in_read[3]),
      .u_out_din(tap_u_out_u_in_din[4]),
      .u_out_full_n(tap_u_out_u_in_full_n[4]),
      .u_out_write(tap_u_out_u_in_write[4]));
  tap_r0 u_tap_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_out_din(lane_c_in_bind_din[4]),
      .c_out_full_n(lane_c_in_bind_full_n[4]),
      .c_out_write(lane_c_in_bind_write[4]),
      .u_in_dout(tap_u_out_u_in_dout[4]),
      .u_in_empty_n(tap_u_out_u_in_empty_n[4]),
      .u_in_read(tap_u_out_u_in_read[4]),
      .u_out_din(tap_u_out_u_in_din[5]),
      .u_out_full_n(tap_u_out_u_in_full_n[5]),
      .u_out_write(tap_u_out_u_in_write[5]));
  tap_r0 u_tap_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_out_din(lane_c_in_bind_din[5]),
      .c_out_full_n(lane_c_in_bind_full_n[5]),
      .c_out_write(lane_c_in_bind_write[5]),
      .u_in_dout(tap_u_out_u_in_dout[5]),
      .u_in_empty_n(tap_u_out_u_in_empty_n[5]),
      .u_in_read(tap_u_out_u_in_read[5]),
      .u_out_din(tap_u_out_u_in_din[6]),
      .u_out_full_n(tap_u_out_u_in_full_n[6]),
      .u_out_write(tap_u_out_u_in_write[6]));
  tap_r0 u_tap_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_out_din(lane_c_in_bind_din[6]),
      .c_out_full_n(lane_c_in_bind_full_n[6]),
      .c_out_write(lane_c_in_bind_write[6]),
      .u_in_dout(tap_u_out_u_in_dout[6]),
      .u_in_empty_n(tap_u_out_u_in_empty_n[6]),
      .u_in_read(tap_u_out_u_in_read[6]),
      .u_out_din(tap_u_out_u_in_din[7]),
      .u_out_full_n(tap_u_out_u_in_full_n[7]),
      .u_out_write(tap_u_out_u_in_write[7]));
  // role tap_r1: 1 instance(s)
  tap_r1 u_tap_r1_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_out_din(lane_c_in_bind_din[7]),
      .c_out_full_n(lane_c_in_bind_full_n[7]),
      .c_out_write(lane_c_in_bind_write[7]),
      .u_in_dout(tap_u_out_u_in_dout[7]),
      .u_in_empty_n(tap_u_out_u_in_empty_n[7]),
      .u_in_read(tap_u_out_u_in_read[7]));
  // role tap_r2: 1 instance(s)
  tap_r2 u_tap_r2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_out_din(lane_c_in_bind_din[0]),
      .c_out_full_n(lane_c_in_bind_full_n[0]),
      .c_out_write(lane_c_in_bind_write[0]),
      .u_in_dout(tap_u_in_bind_dout[0]),
      .u_in_empty_n(tap_u_in_bind_empty_n[0]),
      .u_in_read(tap_u_in_bind_read[0]),
      .u_out_din(tap_u_out_u_in_din[1]),
      .u_out_full_n(tap_u_out_u_in_full_n[1]),
      .u_out_write(tap_u_out_u_in_write[1]));
  // coordinate axis 0: 8 constant source(s)
  wire [31:0] lane_pid0_dout [0:7];
  wire lane_pid0_empty_n [0:7];
  wire lane_pid0_read [0:7];
  spmw_const #(.DW(32), .VAL(0)) u_lane_pid0_0 (.dout(lane_pid0_dout[0]), .empty_n(lane_pid0_empty_n[0]), .read(lane_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_lane_pid0_1 (.dout(lane_pid0_dout[1]), .empty_n(lane_pid0_empty_n[1]), .read(lane_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_lane_pid0_2 (.dout(lane_pid0_dout[2]), .empty_n(lane_pid0_empty_n[2]), .read(lane_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_lane_pid0_3 (.dout(lane_pid0_dout[3]), .empty_n(lane_pid0_empty_n[3]), .read(lane_pid0_read[3]));
  spmw_const #(.DW(32), .VAL(4)) u_lane_pid0_4 (.dout(lane_pid0_dout[4]), .empty_n(lane_pid0_empty_n[4]), .read(lane_pid0_read[4]));
  spmw_const #(.DW(32), .VAL(5)) u_lane_pid0_5 (.dout(lane_pid0_dout[5]), .empty_n(lane_pid0_empty_n[5]), .read(lane_pid0_read[5]));
  spmw_const #(.DW(32), .VAL(6)) u_lane_pid0_6 (.dout(lane_pid0_dout[6]), .empty_n(lane_pid0_empty_n[6]), .read(lane_pid0_read[6]));
  spmw_const #(.DW(32), .VAL(7)) u_lane_pid0_7 (.dout(lane_pid0_dout[7]), .empty_n(lane_pid0_empty_n[7]), .read(lane_pid0_read[7]));
  // role lane_r0: 8 instance(s)
  lane_r0 u_lane_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(lane_c_in_bind_dout[0]),
      .c_in_empty_n(lane_c_in_bind_empty_n[0]),
      .c_in_read(lane_c_in_bind_read[0]),
      .y_out_din(ctap_y_in_bind_din[0]),
      .y_out_full_n(ctap_y_in_bind_full_n[0]),
      .y_out_write(ctap_y_in_bind_write[0]),
      .z_in_dout(lane_z_in_bind_dout[0]),
      .z_in_empty_n(lane_z_in_bind_empty_n[0]),
      .z_in_read(lane_z_in_bind_read[0]),
      ._pid0_dout(lane_pid0_dout[0]),
      ._pid0_empty_n(lane_pid0_empty_n[0]),
      ._pid0_read(lane_pid0_read[0]));
  lane_r0 u_lane_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(lane_c_in_bind_dout[1]),
      .c_in_empty_n(lane_c_in_bind_empty_n[1]),
      .c_in_read(lane_c_in_bind_read[1]),
      .y_out_din(ctap_y_in_bind_din[1]),
      .y_out_full_n(ctap_y_in_bind_full_n[1]),
      .y_out_write(ctap_y_in_bind_write[1]),
      .z_in_dout(lane_z_in_bind_dout[1]),
      .z_in_empty_n(lane_z_in_bind_empty_n[1]),
      .z_in_read(lane_z_in_bind_read[1]),
      ._pid0_dout(lane_pid0_dout[1]),
      ._pid0_empty_n(lane_pid0_empty_n[1]),
      ._pid0_read(lane_pid0_read[1]));
  lane_r0 u_lane_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(lane_c_in_bind_dout[2]),
      .c_in_empty_n(lane_c_in_bind_empty_n[2]),
      .c_in_read(lane_c_in_bind_read[2]),
      .y_out_din(ctap_y_in_bind_din[2]),
      .y_out_full_n(ctap_y_in_bind_full_n[2]),
      .y_out_write(ctap_y_in_bind_write[2]),
      .z_in_dout(lane_z_in_bind_dout[2]),
      .z_in_empty_n(lane_z_in_bind_empty_n[2]),
      .z_in_read(lane_z_in_bind_read[2]),
      ._pid0_dout(lane_pid0_dout[2]),
      ._pid0_empty_n(lane_pid0_empty_n[2]),
      ._pid0_read(lane_pid0_read[2]));
  lane_r0 u_lane_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(lane_c_in_bind_dout[3]),
      .c_in_empty_n(lane_c_in_bind_empty_n[3]),
      .c_in_read(lane_c_in_bind_read[3]),
      .y_out_din(ctap_y_in_bind_din[3]),
      .y_out_full_n(ctap_y_in_bind_full_n[3]),
      .y_out_write(ctap_y_in_bind_write[3]),
      .z_in_dout(lane_z_in_bind_dout[3]),
      .z_in_empty_n(lane_z_in_bind_empty_n[3]),
      .z_in_read(lane_z_in_bind_read[3]),
      ._pid0_dout(lane_pid0_dout[3]),
      ._pid0_empty_n(lane_pid0_empty_n[3]),
      ._pid0_read(lane_pid0_read[3]));
  lane_r0 u_lane_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(lane_c_in_bind_dout[4]),
      .c_in_empty_n(lane_c_in_bind_empty_n[4]),
      .c_in_read(lane_c_in_bind_read[4]),
      .y_out_din(ctap_y_in_bind_din[4]),
      .y_out_full_n(ctap_y_in_bind_full_n[4]),
      .y_out_write(ctap_y_in_bind_write[4]),
      .z_in_dout(lane_z_in_bind_dout[4]),
      .z_in_empty_n(lane_z_in_bind_empty_n[4]),
      .z_in_read(lane_z_in_bind_read[4]),
      ._pid0_dout(lane_pid0_dout[4]),
      ._pid0_empty_n(lane_pid0_empty_n[4]),
      ._pid0_read(lane_pid0_read[4]));
  lane_r0 u_lane_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(lane_c_in_bind_dout[5]),
      .c_in_empty_n(lane_c_in_bind_empty_n[5]),
      .c_in_read(lane_c_in_bind_read[5]),
      .y_out_din(ctap_y_in_bind_din[5]),
      .y_out_full_n(ctap_y_in_bind_full_n[5]),
      .y_out_write(ctap_y_in_bind_write[5]),
      .z_in_dout(lane_z_in_bind_dout[5]),
      .z_in_empty_n(lane_z_in_bind_empty_n[5]),
      .z_in_read(lane_z_in_bind_read[5]),
      ._pid0_dout(lane_pid0_dout[5]),
      ._pid0_empty_n(lane_pid0_empty_n[5]),
      ._pid0_read(lane_pid0_read[5]));
  lane_r0 u_lane_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(lane_c_in_bind_dout[6]),
      .c_in_empty_n(lane_c_in_bind_empty_n[6]),
      .c_in_read(lane_c_in_bind_read[6]),
      .y_out_din(ctap_y_in_bind_din[6]),
      .y_out_full_n(ctap_y_in_bind_full_n[6]),
      .y_out_write(ctap_y_in_bind_write[6]),
      .z_in_dout(lane_z_in_bind_dout[6]),
      .z_in_empty_n(lane_z_in_bind_empty_n[6]),
      .z_in_read(lane_z_in_bind_read[6]),
      ._pid0_dout(lane_pid0_dout[6]),
      ._pid0_empty_n(lane_pid0_empty_n[6]),
      ._pid0_read(lane_pid0_read[6]));
  lane_r0 u_lane_r0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_in_dout(lane_c_in_bind_dout[7]),
      .c_in_empty_n(lane_c_in_bind_empty_n[7]),
      .c_in_read(lane_c_in_bind_read[7]),
      .y_out_din(ctap_y_in_bind_din[7]),
      .y_out_full_n(ctap_y_in_bind_full_n[7]),
      .y_out_write(ctap_y_in_bind_write[7]),
      .z_in_dout(lane_z_in_bind_dout[7]),
      .z_in_empty_n(lane_z_in_bind_empty_n[7]),
      .z_in_read(lane_z_in_bind_read[7]),
      ._pid0_dout(lane_pid0_dout[7]),
      ._pid0_empty_n(lane_pid0_empty_n[7]),
      ._pid0_read(lane_pid0_read[7]));
  // coordinate axis 0: 8 constant source(s)
  wire [31:0] ctap_pid0_dout [0:7];
  wire ctap_pid0_empty_n [0:7];
  wire ctap_pid0_read [0:7];
  spmw_const #(.DW(32), .VAL(0)) u_ctap_pid0_0 (.dout(ctap_pid0_dout[0]), .empty_n(ctap_pid0_empty_n[0]), .read(ctap_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_ctap_pid0_1 (.dout(ctap_pid0_dout[1]), .empty_n(ctap_pid0_empty_n[1]), .read(ctap_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_ctap_pid0_2 (.dout(ctap_pid0_dout[2]), .empty_n(ctap_pid0_empty_n[2]), .read(ctap_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_ctap_pid0_3 (.dout(ctap_pid0_dout[3]), .empty_n(ctap_pid0_empty_n[3]), .read(ctap_pid0_read[3]));
  spmw_const #(.DW(32), .VAL(4)) u_ctap_pid0_4 (.dout(ctap_pid0_dout[4]), .empty_n(ctap_pid0_empty_n[4]), .read(ctap_pid0_read[4]));
  spmw_const #(.DW(32), .VAL(5)) u_ctap_pid0_5 (.dout(ctap_pid0_dout[5]), .empty_n(ctap_pid0_empty_n[5]), .read(ctap_pid0_read[5]));
  spmw_const #(.DW(32), .VAL(6)) u_ctap_pid0_6 (.dout(ctap_pid0_dout[6]), .empty_n(ctap_pid0_empty_n[6]), .read(ctap_pid0_read[6]));
  spmw_const #(.DW(32), .VAL(7)) u_ctap_pid0_7 (.dout(ctap_pid0_dout[7]), .empty_n(ctap_pid0_empty_n[7]), .read(ctap_pid0_read[7]));
  // role ctap_r0: 6 instance(s)
  ctap_r0 u_ctap_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .r_in_dout(ctap_r_out_r_in_dout[1]),
      .r_in_empty_n(ctap_r_out_r_in_empty_n[1]),
      .r_in_read(ctap_r_out_r_in_read[1]),
      .r_out_din(ctap_r_out_r_in_din[2]),
      .r_out_full_n(ctap_r_out_r_in_full_n[2]),
      .r_out_write(ctap_r_out_r_in_write[2]),
      .y_in_dout(ctap_y_in_bind_dout[1]),
      .y_in_empty_n(ctap_y_in_bind_empty_n[1]),
      .y_in_read(ctap_y_in_bind_read[1]),
      ._pid0_dout(ctap_pid0_dout[1]),
      ._pid0_empty_n(ctap_pid0_empty_n[1]),
      ._pid0_read(ctap_pid0_read[1]));
  ctap_r0 u_ctap_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .r_in_dout(ctap_r_out_r_in_dout[2]),
      .r_in_empty_n(ctap_r_out_r_in_empty_n[2]),
      .r_in_read(ctap_r_out_r_in_read[2]),
      .r_out_din(ctap_r_out_r_in_din[3]),
      .r_out_full_n(ctap_r_out_r_in_full_n[3]),
      .r_out_write(ctap_r_out_r_in_write[3]),
      .y_in_dout(ctap_y_in_bind_dout[2]),
      .y_in_empty_n(ctap_y_in_bind_empty_n[2]),
      .y_in_read(ctap_y_in_bind_read[2]),
      ._pid0_dout(ctap_pid0_dout[2]),
      ._pid0_empty_n(ctap_pid0_empty_n[2]),
      ._pid0_read(ctap_pid0_read[2]));
  ctap_r0 u_ctap_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .r_in_dout(ctap_r_out_r_in_dout[3]),
      .r_in_empty_n(ctap_r_out_r_in_empty_n[3]),
      .r_in_read(ctap_r_out_r_in_read[3]),
      .r_out_din(ctap_r_out_r_in_din[4]),
      .r_out_full_n(ctap_r_out_r_in_full_n[4]),
      .r_out_write(ctap_r_out_r_in_write[4]),
      .y_in_dout(ctap_y_in_bind_dout[3]),
      .y_in_empty_n(ctap_y_in_bind_empty_n[3]),
      .y_in_read(ctap_y_in_bind_read[3]),
      ._pid0_dout(ctap_pid0_dout[3]),
      ._pid0_empty_n(ctap_pid0_empty_n[3]),
      ._pid0_read(ctap_pid0_read[3]));
  ctap_r0 u_ctap_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .r_in_dout(ctap_r_out_r_in_dout[4]),
      .r_in_empty_n(ctap_r_out_r_in_empty_n[4]),
      .r_in_read(ctap_r_out_r_in_read[4]),
      .r_out_din(ctap_r_out_r_in_din[5]),
      .r_out_full_n(ctap_r_out_r_in_full_n[5]),
      .r_out_write(ctap_r_out_r_in_write[5]),
      .y_in_dout(ctap_y_in_bind_dout[4]),
      .y_in_empty_n(ctap_y_in_bind_empty_n[4]),
      .y_in_read(ctap_y_in_bind_read[4]),
      ._pid0_dout(ctap_pid0_dout[4]),
      ._pid0_empty_n(ctap_pid0_empty_n[4]),
      ._pid0_read(ctap_pid0_read[4]));
  ctap_r0 u_ctap_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .r_in_dout(ctap_r_out_r_in_dout[5]),
      .r_in_empty_n(ctap_r_out_r_in_empty_n[5]),
      .r_in_read(ctap_r_out_r_in_read[5]),
      .r_out_din(ctap_r_out_r_in_din[6]),
      .r_out_full_n(ctap_r_out_r_in_full_n[6]),
      .r_out_write(ctap_r_out_r_in_write[6]),
      .y_in_dout(ctap_y_in_bind_dout[5]),
      .y_in_empty_n(ctap_y_in_bind_empty_n[5]),
      .y_in_read(ctap_y_in_bind_read[5]),
      ._pid0_dout(ctap_pid0_dout[5]),
      ._pid0_empty_n(ctap_pid0_empty_n[5]),
      ._pid0_read(ctap_pid0_read[5]));
  ctap_r0 u_ctap_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .r_in_dout(ctap_r_out_r_in_dout[6]),
      .r_in_empty_n(ctap_r_out_r_in_empty_n[6]),
      .r_in_read(ctap_r_out_r_in_read[6]),
      .r_out_din(ctap_r_out_r_in_din[7]),
      .r_out_full_n(ctap_r_out_r_in_full_n[7]),
      .r_out_write(ctap_r_out_r_in_write[7]),
      .y_in_dout(ctap_y_in_bind_dout[6]),
      .y_in_empty_n(ctap_y_in_bind_empty_n[6]),
      .y_in_read(ctap_y_in_bind_read[6]),
      ._pid0_dout(ctap_pid0_dout[6]),
      ._pid0_empty_n(ctap_pid0_empty_n[6]),
      ._pid0_read(ctap_pid0_read[6]));
  // role ctap_r1: 1 instance(s)
  ctap_r1 u_ctap_r1_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .r_in_dout(ctap_r_out_r_in_dout[7]),
      .r_in_empty_n(ctap_r_out_r_in_empty_n[7]),
      .r_in_read(ctap_r_out_r_in_read[7]),
      .r_out_din(pack8_row_in_bind_din[0]),
      .r_out_full_n(pack8_row_in_bind_full_n[0]),
      .r_out_write(pack8_row_in_bind_write[0]),
      .y_in_dout(ctap_y_in_bind_dout[7]),
      .y_in_empty_n(ctap_y_in_bind_empty_n[7]),
      .y_in_read(ctap_y_in_bind_read[7]),
      ._pid0_dout(ctap_pid0_dout[7]),
      ._pid0_empty_n(ctap_pid0_empty_n[7]),
      ._pid0_read(ctap_pid0_read[7]));
  // role ctap_r2: 1 instance(s)
  ctap_r2 u_ctap_r2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .r_out_din(ctap_r_out_r_in_din[1]),
      .r_out_full_n(ctap_r_out_r_in_full_n[1]),
      .r_out_write(ctap_r_out_r_in_write[1]),
      .y_in_dout(ctap_y_in_bind_dout[0]),
      .y_in_empty_n(ctap_y_in_bind_empty_n[0]),
      .y_in_read(ctap_y_in_bind_read[0]),
      ._pid0_dout(ctap_pid0_dout[0]),
      ._pid0_empty_n(ctap_pid0_empty_n[0]),
      ._pid0_read(ctap_pid0_read[0]));
  // role wreq8_r0: 1 instance(s)
  wreq8_r0 u_wreq8_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .done_din(wreq8_done_bind_din[0]),
      .done_full_n(wreq8_done_bind_full_n[0]),
      .done_write(wreq8_done_bind_write[0]),
      .wr_ack_dout(wreq8_wr_ack_bind_dout[0]),
      .wr_ack_empty_n(wreq8_wr_ack_bind_empty_n[0]),
      .wr_ack_read(wreq8_wr_ack_bind_read[0]),
      .wr_cmd_din(wreq8_wr_cmd_bind_din[0]),
      .wr_cmd_full_n(wreq8_wr_cmd_bind_full_n[0]),
      .wr_cmd_write(wreq8_wr_cmd_bind_write[0]),
      .y_cmd_dout(wreq8_y_cmd_bind_dout[0]),
      .y_cmd_empty_n(wreq8_y_cmd_bind_empty_n[0]),
      .y_cmd_read(wreq8_y_cmd_bind_read[0]));
  // role pack8_r0: 1 instance(s)
  pack8_r0 u_pack8_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .credit_din(head8_credit_bind_din[0]),
      .credit_full_n(head8_credit_bind_full_n[0]),
      .credit_write(head8_credit_bind_write[0]),
      .row_in_dout(pack8_row_in_bind_dout[0]),
      .row_in_empty_n(pack8_row_in_bind_empty_n[0]),
      .row_in_read(pack8_row_in_bind_read[0]),
      .wr_data_din(pack8_wr_data_bind_din[0]),
      .wr_data_full_n(pack8_wr_data_bind_full_n[0]),
      .wr_data_write(pack8_wr_data_bind_write[0]));
endmodule
