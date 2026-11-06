`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] req4_launch_bind_dout [0:0],
  input  wire req4_launch_bind_empty_n [0:0],
  output wire req4_launch_bind_read [0:0],
  output wire [63:0] req4_rd_cmd_bind_din [0:0],
  output wire req4_rd_cmd_bind_write [0:0],
  input  wire req4_rd_cmd_bind_full_n [0:0],
  input  wire [63:0] deal4_rd_data_bind_dout [0:0],
  input  wire deal4_rd_data_bind_empty_n [0:0],
  output wire deal4_rd_data_bind_read [0:0],
  input  wire [63:0] wreq4_wr_ack_bind_dout [0:0],
  input  wire wreq4_wr_ack_bind_empty_n [0:0],
  output wire wreq4_wr_ack_bind_read [0:0],
  output wire [63:0] wreq4_wr_cmd_bind_din [0:0],
  output wire wreq4_wr_cmd_bind_write [0:0],
  input  wire wreq4_wr_cmd_bind_full_n [0:0],
  output wire [63:0] wreq4_done_bind_din [0:0],
  output wire wreq4_done_bind_write [0:0],
  input  wire wreq4_done_bind_full_n [0:0],
  output wire [63:0] pack4_wr_data_bind_din [0:0],
  output wire pack4_wr_data_bind_write [0:0],
  input  wire pack4_wr_data_bind_full_n [0:0]
);
  // family pe_a_out_a_in: 16 channel(s), 16-bit, depth 0
  wire [15:0] pe_a_out_a_in_din [0:15];
  wire [15:0] pe_a_out_a_in_dout [0:15];
  wire pe_a_out_a_in_full_n [0:15];
  wire pe_a_out_a_in_write [0:15];
  wire pe_a_out_a_in_empty_n [0:15];
  wire pe_a_out_a_in_read [0:15];
  genvar pe_a_out_a_in_i;
  generate
    for (pe_a_out_a_in_i = 0; pe_a_out_a_in_i < 16; pe_a_out_a_in_i = pe_a_out_a_in_i + 1) begin : g_pe_a_out_a_in
      spmw_fifo #(.DW(16), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_a_out_a_in_din[pe_a_out_a_in_i]), .full_n(pe_a_out_a_in_full_n[pe_a_out_a_in_i]), .write(pe_a_out_a_in_write[pe_a_out_a_in_i]), .dout(pe_a_out_a_in_dout[pe_a_out_a_in_i]), .empty_n(pe_a_out_a_in_empty_n[pe_a_out_a_in_i]), .read(pe_a_out_a_in_read[pe_a_out_a_in_i]));
    end
  endgenerate
  // family pe_w_out_w_in: 16 channel(s), 8-bit, depth 0
  wire [7:0] pe_w_out_w_in_din [0:15];
  wire [7:0] pe_w_out_w_in_dout [0:15];
  wire pe_w_out_w_in_full_n [0:15];
  wire pe_w_out_w_in_write [0:15];
  wire pe_w_out_w_in_empty_n [0:15];
  wire pe_w_out_w_in_read [0:15];
  genvar pe_w_out_w_in_i;
  generate
    for (pe_w_out_w_in_i = 0; pe_w_out_w_in_i < 16; pe_w_out_w_in_i = pe_w_out_w_in_i + 1) begin : g_pe_w_out_w_in
      spmw_fifo #(.DW(8), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_w_out_w_in_din[pe_w_out_w_in_i]), .full_n(pe_w_out_w_in_full_n[pe_w_out_w_in_i]), .write(pe_w_out_w_in_write[pe_w_out_w_in_i]), .dout(pe_w_out_w_in_dout[pe_w_out_w_in_i]), .empty_n(pe_w_out_w_in_empty_n[pe_w_out_w_in_i]), .read(pe_w_out_w_in_read[pe_w_out_w_in_i]));
    end
  endgenerate
  // family pe_p_out_p_in: 16 channel(s), 32-bit, depth 0
  wire [31:0] pe_p_out_p_in_din [0:15];
  wire [31:0] pe_p_out_p_in_dout [0:15];
  wire pe_p_out_p_in_full_n [0:15];
  wire pe_p_out_p_in_write [0:15];
  wire pe_p_out_p_in_empty_n [0:15];
  wire pe_p_out_p_in_read [0:15];
  genvar pe_p_out_p_in_i;
  generate
    for (pe_p_out_p_in_i = 0; pe_p_out_p_in_i < 16; pe_p_out_p_in_i = pe_p_out_p_in_i + 1) begin : g_pe_p_out_p_in
      spmw_fifo #(.DW(32), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_p_out_p_in_din[pe_p_out_p_in_i]), .full_n(pe_p_out_p_in_full_n[pe_p_out_p_in_i]), .write(pe_p_out_p_in_write[pe_p_out_p_in_i]), .dout(pe_p_out_p_in_dout[pe_p_out_p_in_i]), .empty_n(pe_p_out_p_in_empty_n[pe_p_out_p_in_i]), .read(pe_p_out_p_in_read[pe_p_out_p_in_i]));
    end
  endgenerate
  // family pe_a_in_bind: 4 channel(s), 16-bit, depth 0
  wire [15:0] pe_a_in_bind_din [0:3];
  wire [15:0] pe_a_in_bind_dout [0:3];
  wire pe_a_in_bind_full_n [0:3];
  wire pe_a_in_bind_write [0:3];
  wire pe_a_in_bind_empty_n [0:3];
  wire pe_a_in_bind_read [0:3];
  genvar pe_a_in_bind_i;
  generate
    for (pe_a_in_bind_i = 0; pe_a_in_bind_i < 4; pe_a_in_bind_i = pe_a_in_bind_i + 1) begin : g_pe_a_in_bind
      spmw_fifo #(.DW(16), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_a_in_bind_din[pe_a_in_bind_i]), .full_n(pe_a_in_bind_full_n[pe_a_in_bind_i]), .write(pe_a_in_bind_write[pe_a_in_bind_i]), .dout(pe_a_in_bind_dout[pe_a_in_bind_i]), .empty_n(pe_a_in_bind_empty_n[pe_a_in_bind_i]), .read(pe_a_in_bind_read[pe_a_in_bind_i]));
    end
  endgenerate
  // family pe_w_in_bind: 4 channel(s), 8-bit, depth 0
  wire [7:0] pe_w_in_bind_din [0:3];
  wire [7:0] pe_w_in_bind_dout [0:3];
  wire pe_w_in_bind_full_n [0:3];
  wire pe_w_in_bind_write [0:3];
  wire pe_w_in_bind_empty_n [0:3];
  wire pe_w_in_bind_read [0:3];
  genvar pe_w_in_bind_i;
  generate
    for (pe_w_in_bind_i = 0; pe_w_in_bind_i < 4; pe_w_in_bind_i = pe_w_in_bind_i + 1) begin : g_pe_w_in_bind
      spmw_fifo #(.DW(8), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_w_in_bind_din[pe_w_in_bind_i]), .full_n(pe_w_in_bind_full_n[pe_w_in_bind_i]), .write(pe_w_in_bind_write[pe_w_in_bind_i]), .dout(pe_w_in_bind_dout[pe_w_in_bind_i]), .empty_n(pe_w_in_bind_empty_n[pe_w_in_bind_i]), .read(pe_w_in_bind_read[pe_w_in_bind_i]));
    end
  endgenerate
  // family lane_z_in_bind: 4 channel(s), 32-bit, depth 2
  wire [31:0] lane_z_in_bind_din [0:3];
  wire [31:0] lane_z_in_bind_dout [0:3];
  wire lane_z_in_bind_full_n [0:3];
  wire lane_z_in_bind_write [0:3];
  wire lane_z_in_bind_empty_n [0:3];
  wire lane_z_in_bind_read [0:3];
  genvar lane_z_in_bind_i;
  generate
    for (lane_z_in_bind_i = 0; lane_z_in_bind_i < 4; lane_z_in_bind_i = lane_z_in_bind_i + 1) begin : g_lane_z_in_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(lane_z_in_bind_din[lane_z_in_bind_i]), .full_n(lane_z_in_bind_full_n[lane_z_in_bind_i]), .write(lane_z_in_bind_write[lane_z_in_bind_i]), .dout(lane_z_in_bind_dout[lane_z_in_bind_i]), .empty_n(lane_z_in_bind_empty_n[lane_z_in_bind_i]), .read(lane_z_in_bind_read[lane_z_in_bind_i]));
    end
  endgenerate
  // family etap_e_out_e_in: 4 channel(s), 72-bit, depth 0
  wire [71:0] etap_e_out_e_in_din [0:3];
  wire [71:0] etap_e_out_e_in_dout [0:3];
  wire etap_e_out_e_in_full_n [0:3];
  wire etap_e_out_e_in_write [0:3];
  wire etap_e_out_e_in_empty_n [0:3];
  wire etap_e_out_e_in_read [0:3];
  genvar etap_e_out_e_in_i;
  generate
    for (etap_e_out_e_in_i = 0; etap_e_out_e_in_i < 4; etap_e_out_e_in_i = etap_e_out_e_in_i + 1) begin : g_etap_e_out_e_in
      spmw_fifo #(.DW(72), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(etap_e_out_e_in_din[etap_e_out_e_in_i]), .full_n(etap_e_out_e_in_full_n[etap_e_out_e_in_i]), .write(etap_e_out_e_in_write[etap_e_out_e_in_i]), .dout(etap_e_out_e_in_dout[etap_e_out_e_in_i]), .empty_n(etap_e_out_e_in_empty_n[etap_e_out_e_in_i]), .read(etap_e_out_e_in_read[etap_e_out_e_in_i]));
    end
  endgenerate
  // family etap_e_in_bind: 1 channel(s), 72-bit, depth 0
  wire [71:0] etap_e_in_bind_din [0:0];
  wire [71:0] etap_e_in_bind_dout [0:0];
  wire etap_e_in_bind_full_n [0:0];
  wire etap_e_in_bind_write [0:0];
  wire etap_e_in_bind_empty_n [0:0];
  wire etap_e_in_bind_read [0:0];
  genvar etap_e_in_bind_i;
  generate
    for (etap_e_in_bind_i = 0; etap_e_in_bind_i < 1; etap_e_in_bind_i = etap_e_in_bind_i + 1) begin : g_etap_e_in_bind
      spmw_fifo #(.DW(72), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(etap_e_in_bind_din[etap_e_in_bind_i]), .full_n(etap_e_in_bind_full_n[etap_e_in_bind_i]), .write(etap_e_in_bind_write[etap_e_in_bind_i]), .dout(etap_e_in_bind_dout[etap_e_in_bind_i]), .empty_n(etap_e_in_bind_empty_n[etap_e_in_bind_i]), .read(etap_e_in_bind_read[etap_e_in_bind_i]));
    end
  endgenerate
  // family deal4_tag_in_bind: 1 channel(s), 32-bit, depth 4
  wire [31:0] deal4_tag_in_bind_din [0:0];
  wire [31:0] deal4_tag_in_bind_dout [0:0];
  wire deal4_tag_in_bind_full_n [0:0];
  wire deal4_tag_in_bind_write [0:0];
  wire deal4_tag_in_bind_empty_n [0:0];
  wire deal4_tag_in_bind_read [0:0];
  genvar deal4_tag_in_bind_i;
  generate
    for (deal4_tag_in_bind_i = 0; deal4_tag_in_bind_i < 1; deal4_tag_in_bind_i = deal4_tag_in_bind_i + 1) begin : g_deal4_tag_in_bind
      spmw_fifo #(.DW(32), .DEPTH(4)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(deal4_tag_in_bind_din[deal4_tag_in_bind_i]), .full_n(deal4_tag_in_bind_full_n[deal4_tag_in_bind_i]), .write(deal4_tag_in_bind_write[deal4_tag_in_bind_i]), .dout(deal4_tag_in_bind_dout[deal4_tag_in_bind_i]), .empty_n(deal4_tag_in_bind_empty_n[deal4_tag_in_bind_i]), .read(deal4_tag_in_bind_read[deal4_tag_in_bind_i]));
    end
  endgenerate
  // family req4_ins_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] req4_ins_in_bind_din [0:0];
  wire [63:0] req4_ins_in_bind_dout [0:0];
  wire req4_ins_in_bind_full_n [0:0];
  wire req4_ins_in_bind_write [0:0];
  wire req4_ins_in_bind_empty_n [0:0];
  wire req4_ins_in_bind_read [0:0];
  genvar req4_ins_in_bind_i;
  generate
    for (req4_ins_in_bind_i = 0; req4_ins_in_bind_i < 1; req4_ins_in_bind_i = req4_ins_in_bind_i + 1) begin : g_req4_ins_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(req4_ins_in_bind_din[req4_ins_in_bind_i]), .full_n(req4_ins_in_bind_full_n[req4_ins_in_bind_i]), .write(req4_ins_in_bind_write[req4_ins_in_bind_i]), .dout(req4_ins_in_bind_dout[req4_ins_in_bind_i]), .empty_n(req4_ins_in_bind_empty_n[req4_ins_in_bind_i]), .read(req4_ins_in_bind_read[req4_ins_in_bind_i]));
    end
  endgenerate
  // family head4_op_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] head4_op_in_bind_din [0:0];
  wire [63:0] head4_op_in_bind_dout [0:0];
  wire head4_op_in_bind_full_n [0:0];
  wire head4_op_in_bind_write [0:0];
  wire head4_op_in_bind_empty_n [0:0];
  wire head4_op_in_bind_read [0:0];
  genvar head4_op_in_bind_i;
  generate
    for (head4_op_in_bind_i = 0; head4_op_in_bind_i < 1; head4_op_in_bind_i = head4_op_in_bind_i + 1) begin : g_head4_op_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(head4_op_in_bind_din[head4_op_in_bind_i]), .full_n(head4_op_in_bind_full_n[head4_op_in_bind_i]), .write(head4_op_in_bind_write[head4_op_in_bind_i]), .dout(head4_op_in_bind_dout[head4_op_in_bind_i]), .empty_n(head4_op_in_bind_empty_n[head4_op_in_bind_i]), .read(head4_op_in_bind_read[head4_op_in_bind_i]));
    end
  endgenerate
  // family wreq4_y_cmd_bind: 1 channel(s), 64-bit, depth 16
  wire [63:0] wreq4_y_cmd_bind_din [0:0];
  wire [63:0] wreq4_y_cmd_bind_dout [0:0];
  wire wreq4_y_cmd_bind_full_n [0:0];
  wire wreq4_y_cmd_bind_write [0:0];
  wire wreq4_y_cmd_bind_empty_n [0:0];
  wire wreq4_y_cmd_bind_read [0:0];
  genvar wreq4_y_cmd_bind_i;
  generate
    for (wreq4_y_cmd_bind_i = 0; wreq4_y_cmd_bind_i < 1; wreq4_y_cmd_bind_i = wreq4_y_cmd_bind_i + 1) begin : g_wreq4_y_cmd_bind
      spmw_fifo #(.DW(64), .DEPTH(16)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(wreq4_y_cmd_bind_din[wreq4_y_cmd_bind_i]), .full_n(wreq4_y_cmd_bind_full_n[wreq4_y_cmd_bind_i]), .write(wreq4_y_cmd_bind_write[wreq4_y_cmd_bind_i]), .dout(wreq4_y_cmd_bind_dout[wreq4_y_cmd_bind_i]), .empty_n(wreq4_y_cmd_bind_empty_n[wreq4_y_cmd_bind_i]), .read(wreq4_y_cmd_bind_read[wreq4_y_cmd_bind_i]));
    end
  endgenerate
  // family head4_a_in_bind: 1 channel(s), 32-bit, depth 128
  wire [31:0] head4_a_in_bind_din [0:0];
  wire [31:0] head4_a_in_bind_dout [0:0];
  wire head4_a_in_bind_full_n [0:0];
  wire head4_a_in_bind_write [0:0];
  wire head4_a_in_bind_empty_n [0:0];
  wire head4_a_in_bind_read [0:0];
  genvar head4_a_in_bind_i;
  generate
    for (head4_a_in_bind_i = 0; head4_a_in_bind_i < 1; head4_a_in_bind_i = head4_a_in_bind_i + 1) begin : g_head4_a_in_bind
      spmw_fifo_bram #(.DW(32), .DEPTH(128)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(head4_a_in_bind_din[head4_a_in_bind_i]), .full_n(head4_a_in_bind_full_n[head4_a_in_bind_i]), .write(head4_a_in_bind_write[head4_a_in_bind_i]), .dout(head4_a_in_bind_dout[head4_a_in_bind_i]), .empty_n(head4_a_in_bind_empty_n[head4_a_in_bind_i]), .read(head4_a_in_bind_read[head4_a_in_bind_i]));
    end
  endgenerate
  // family head4_w_in_bind: 1 channel(s), 32-bit, depth 256
  wire [31:0] head4_w_in_bind_din [0:0];
  wire [31:0] head4_w_in_bind_dout [0:0];
  wire head4_w_in_bind_full_n [0:0];
  wire head4_w_in_bind_write [0:0];
  wire head4_w_in_bind_empty_n [0:0];
  wire head4_w_in_bind_read [0:0];
  genvar head4_w_in_bind_i;
  generate
    for (head4_w_in_bind_i = 0; head4_w_in_bind_i < 1; head4_w_in_bind_i = head4_w_in_bind_i + 1) begin : g_head4_w_in_bind
      spmw_fifo_bram #(.DW(32), .DEPTH(256)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(head4_w_in_bind_din[head4_w_in_bind_i]), .full_n(head4_w_in_bind_full_n[head4_w_in_bind_i]), .write(head4_w_in_bind_write[head4_w_in_bind_i]), .dout(head4_w_in_bind_dout[head4_w_in_bind_i]), .empty_n(head4_w_in_bind_empty_n[head4_w_in_bind_i]), .read(head4_w_in_bind_read[head4_w_in_bind_i]));
    end
  endgenerate
  // family head4_b_in_bind: 1 channel(s), 64-bit, depth 16
  wire [63:0] head4_b_in_bind_din [0:0];
  wire [63:0] head4_b_in_bind_dout [0:0];
  wire head4_b_in_bind_full_n [0:0];
  wire head4_b_in_bind_write [0:0];
  wire head4_b_in_bind_empty_n [0:0];
  wire head4_b_in_bind_read [0:0];
  genvar head4_b_in_bind_i;
  generate
    for (head4_b_in_bind_i = 0; head4_b_in_bind_i < 1; head4_b_in_bind_i = head4_b_in_bind_i + 1) begin : g_head4_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(16)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(head4_b_in_bind_din[head4_b_in_bind_i]), .full_n(head4_b_in_bind_full_n[head4_b_in_bind_i]), .write(head4_b_in_bind_write[head4_b_in_bind_i]), .dout(head4_b_in_bind_dout[head4_b_in_bind_i]), .empty_n(head4_b_in_bind_empty_n[head4_b_in_bind_i]), .read(head4_b_in_bind_read[head4_b_in_bind_i]));
    end
  endgenerate
  // family uq_u_in_bind: 1 channel(s), 64-bit, depth 20
  wire [63:0] uq_u_in_bind_din [0:0];
  wire [63:0] uq_u_in_bind_dout [0:0];
  wire uq_u_in_bind_full_n [0:0];
  wire uq_u_in_bind_write [0:0];
  wire uq_u_in_bind_empty_n [0:0];
  wire uq_u_in_bind_read [0:0];
  genvar uq_u_in_bind_i;
  generate
    for (uq_u_in_bind_i = 0; uq_u_in_bind_i < 1; uq_u_in_bind_i = uq_u_in_bind_i + 1) begin : g_uq_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(20)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(uq_u_in_bind_din[uq_u_in_bind_i]), .full_n(uq_u_in_bind_full_n[uq_u_in_bind_i]), .write(uq_u_in_bind_write[uq_u_in_bind_i]), .dout(uq_u_in_bind_dout[uq_u_in_bind_i]), .empty_n(uq_u_in_bind_empty_n[uq_u_in_bind_i]), .read(uq_u_in_bind_read[uq_u_in_bind_i]));
    end
  endgenerate
  // family head4_credit_bind: 1 channel(s), 8-bit, depth 2048
  wire [7:0] head4_credit_bind_din [0:0];
  wire [7:0] head4_credit_bind_dout [0:0];
  wire head4_credit_bind_full_n [0:0];
  wire head4_credit_bind_write [0:0];
  wire head4_credit_bind_empty_n [0:0];
  wire head4_credit_bind_read [0:0];
  genvar head4_credit_bind_i;
  generate
    for (head4_credit_bind_i = 0; head4_credit_bind_i < 1; head4_credit_bind_i = head4_credit_bind_i + 1) begin : g_head4_credit_bind
      spmw_fifo_bram #(.DW(8), .DEPTH(2048)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(head4_credit_bind_din[head4_credit_bind_i]), .full_n(head4_credit_bind_full_n[head4_credit_bind_i]), .write(head4_credit_bind_write[head4_credit_bind_i]), .dout(head4_credit_bind_dout[head4_credit_bind_i]), .empty_n(head4_credit_bind_empty_n[head4_credit_bind_i]), .read(head4_credit_bind_read[head4_credit_bind_i]));
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
  // family tap_u_out_u_in: 4 channel(s), 64-bit, depth 2
  wire [63:0] tap_u_out_u_in_din [0:3];
  wire [63:0] tap_u_out_u_in_dout [0:3];
  wire tap_u_out_u_in_full_n [0:3];
  wire tap_u_out_u_in_write [0:3];
  wire tap_u_out_u_in_empty_n [0:3];
  wire tap_u_out_u_in_read [0:3];
  genvar tap_u_out_u_in_i;
  generate
    for (tap_u_out_u_in_i = 0; tap_u_out_u_in_i < 4; tap_u_out_u_in_i = tap_u_out_u_in_i + 1) begin : g_tap_u_out_u_in
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tap_u_out_u_in_din[tap_u_out_u_in_i]), .full_n(tap_u_out_u_in_full_n[tap_u_out_u_in_i]), .write(tap_u_out_u_in_write[tap_u_out_u_in_i]), .dout(tap_u_out_u_in_dout[tap_u_out_u_in_i]), .empty_n(tap_u_out_u_in_empty_n[tap_u_out_u_in_i]), .read(tap_u_out_u_in_read[tap_u_out_u_in_i]));
    end
  endgenerate
  // family lane_c_in_bind: 4 channel(s), 64-bit, depth 2
  wire [63:0] lane_c_in_bind_din [0:3];
  wire [63:0] lane_c_in_bind_dout [0:3];
  wire lane_c_in_bind_full_n [0:3];
  wire lane_c_in_bind_write [0:3];
  wire lane_c_in_bind_empty_n [0:3];
  wire lane_c_in_bind_read [0:3];
  genvar lane_c_in_bind_i;
  generate
    for (lane_c_in_bind_i = 0; lane_c_in_bind_i < 4; lane_c_in_bind_i = lane_c_in_bind_i + 1) begin : g_lane_c_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(lane_c_in_bind_din[lane_c_in_bind_i]), .full_n(lane_c_in_bind_full_n[lane_c_in_bind_i]), .write(lane_c_in_bind_write[lane_c_in_bind_i]), .dout(lane_c_in_bind_dout[lane_c_in_bind_i]), .empty_n(lane_c_in_bind_empty_n[lane_c_in_bind_i]), .read(lane_c_in_bind_read[lane_c_in_bind_i]));
    end
  endgenerate
  // family ctap_y_in_bind: 4 channel(s), 16-bit, depth 2
  wire [15:0] ctap_y_in_bind_din [0:3];
  wire [15:0] ctap_y_in_bind_dout [0:3];
  wire ctap_y_in_bind_full_n [0:3];
  wire ctap_y_in_bind_write [0:3];
  wire ctap_y_in_bind_empty_n [0:3];
  wire ctap_y_in_bind_read [0:3];
  genvar ctap_y_in_bind_i;
  generate
    for (ctap_y_in_bind_i = 0; ctap_y_in_bind_i < 4; ctap_y_in_bind_i = ctap_y_in_bind_i + 1) begin : g_ctap_y_in_bind
      spmw_fifo #(.DW(16), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(ctap_y_in_bind_din[ctap_y_in_bind_i]), .full_n(ctap_y_in_bind_full_n[ctap_y_in_bind_i]), .write(ctap_y_in_bind_write[ctap_y_in_bind_i]), .dout(ctap_y_in_bind_dout[ctap_y_in_bind_i]), .empty_n(ctap_y_in_bind_empty_n[ctap_y_in_bind_i]), .read(ctap_y_in_bind_read[ctap_y_in_bind_i]));
    end
  endgenerate
  // family ctap_r_out_r_in: 4 channel(s), 40-bit, depth 2
  wire [39:0] ctap_r_out_r_in_din [0:3];
  wire [39:0] ctap_r_out_r_in_dout [0:3];
  wire ctap_r_out_r_in_full_n [0:3];
  wire ctap_r_out_r_in_write [0:3];
  wire ctap_r_out_r_in_empty_n [0:3];
  wire ctap_r_out_r_in_read [0:3];
  genvar ctap_r_out_r_in_i;
  generate
    for (ctap_r_out_r_in_i = 0; ctap_r_out_r_in_i < 4; ctap_r_out_r_in_i = ctap_r_out_r_in_i + 1) begin : g_ctap_r_out_r_in
      spmw_fifo #(.DW(40), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(ctap_r_out_r_in_din[ctap_r_out_r_in_i]), .full_n(ctap_r_out_r_in_full_n[ctap_r_out_r_in_i]), .write(ctap_r_out_r_in_write[ctap_r_out_r_in_i]), .dout(ctap_r_out_r_in_dout[ctap_r_out_r_in_i]), .empty_n(ctap_r_out_r_in_empty_n[ctap_r_out_r_in_i]), .read(ctap_r_out_r_in_read[ctap_r_out_r_in_i]));
    end
  endgenerate
  // family pack4_row_in_bind: 1 channel(s), 40-bit, depth 2048
  wire [39:0] pack4_row_in_bind_din [0:0];
  wire [39:0] pack4_row_in_bind_dout [0:0];
  wire pack4_row_in_bind_full_n [0:0];
  wire pack4_row_in_bind_write [0:0];
  wire pack4_row_in_bind_empty_n [0:0];
  wire pack4_row_in_bind_read [0:0];
  genvar pack4_row_in_bind_i;
  generate
    for (pack4_row_in_bind_i = 0; pack4_row_in_bind_i < 1; pack4_row_in_bind_i = pack4_row_in_bind_i + 1) begin : g_pack4_row_in_bind
      spmw_fifo_bram #(.DW(40), .DEPTH(2048)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pack4_row_in_bind_din[pack4_row_in_bind_i]), .full_n(pack4_row_in_bind_full_n[pack4_row_in_bind_i]), .write(pack4_row_in_bind_write[pack4_row_in_bind_i]), .dout(pack4_row_in_bind_dout[pack4_row_in_bind_i]), .empty_n(pack4_row_in_bind_empty_n[pack4_row_in_bind_i]), .read(pack4_row_in_bind_read[pack4_row_in_bind_i]));
    end
  endgenerate
  // role pe_r0: 4 instance(s)
  pe_r0 u_pe_r0_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[5]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[5]),
      .a_in_read(pe_a_out_a_in_read[5]),
      .a_out_din(pe_a_out_a_in_din[6]),
      .a_out_full_n(pe_a_out_a_in_full_n[6]),
      .a_out_write(pe_a_out_a_in_write[6]),
      .p_in_dout(pe_p_out_p_in_dout[5]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[5]),
      .p_in_read(pe_p_out_p_in_read[5]),
      .p_out_din(pe_p_out_p_in_din[9]),
      .p_out_full_n(pe_p_out_p_in_full_n[9]),
      .p_out_write(pe_p_out_p_in_write[9]),
      .w_in_dout(pe_w_out_w_in_dout[5]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[5]),
      .w_in_read(pe_w_out_w_in_read[5]),
      .w_out_din(pe_w_out_w_in_din[6]),
      .w_out_full_n(pe_w_out_w_in_full_n[6]),
      .w_out_write(pe_w_out_w_in_write[6]));
  pe_r0 u_pe_r0_1_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[6]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[6]),
      .a_in_read(pe_a_out_a_in_read[6]),
      .a_out_din(pe_a_out_a_in_din[7]),
      .a_out_full_n(pe_a_out_a_in_full_n[7]),
      .a_out_write(pe_a_out_a_in_write[7]),
      .p_in_dout(pe_p_out_p_in_dout[6]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[6]),
      .p_in_read(pe_p_out_p_in_read[6]),
      .p_out_din(pe_p_out_p_in_din[10]),
      .p_out_full_n(pe_p_out_p_in_full_n[10]),
      .p_out_write(pe_p_out_p_in_write[10]),
      .w_in_dout(pe_w_out_w_in_dout[6]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[6]),
      .w_in_read(pe_w_out_w_in_read[6]),
      .w_out_din(pe_w_out_w_in_din[7]),
      .w_out_full_n(pe_w_out_w_in_full_n[7]),
      .w_out_write(pe_w_out_w_in_write[7]));
  pe_r0 u_pe_r0_2_1 (
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
      .p_out_din(pe_p_out_p_in_din[13]),
      .p_out_full_n(pe_p_out_p_in_full_n[13]),
      .p_out_write(pe_p_out_p_in_write[13]),
      .w_in_dout(pe_w_out_w_in_dout[9]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[9]),
      .w_in_read(pe_w_out_w_in_read[9]),
      .w_out_din(pe_w_out_w_in_din[10]),
      .w_out_full_n(pe_w_out_w_in_full_n[10]),
      .w_out_write(pe_w_out_w_in_write[10]));
  pe_r0 u_pe_r0_2_2 (
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
      .p_out_din(pe_p_out_p_in_din[14]),
      .p_out_full_n(pe_p_out_p_in_full_n[14]),
      .p_out_write(pe_p_out_p_in_write[14]),
      .w_in_dout(pe_w_out_w_in_dout[10]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[10]),
      .w_in_read(pe_w_out_w_in_read[10]),
      .w_out_din(pe_w_out_w_in_din[11]),
      .w_out_full_n(pe_w_out_w_in_full_n[11]),
      .w_out_write(pe_w_out_w_in_write[11]));
  // role pe_r1: 2 instance(s)
  pe_r1 u_pe_r1_3_1 (
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
      .p_out_din(lane_z_in_bind_din[1]),
      .p_out_full_n(lane_z_in_bind_full_n[1]),
      .p_out_write(lane_z_in_bind_write[1]),
      .w_in_dout(pe_w_out_w_in_dout[13]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[13]),
      .w_in_read(pe_w_out_w_in_read[13]),
      .w_out_din(pe_w_out_w_in_din[14]),
      .w_out_full_n(pe_w_out_w_in_full_n[14]),
      .w_out_write(pe_w_out_w_in_write[14]));
  pe_r1 u_pe_r1_3_2 (
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
      .p_out_din(lane_z_in_bind_din[2]),
      .p_out_full_n(lane_z_in_bind_full_n[2]),
      .p_out_write(lane_z_in_bind_write[2]),
      .w_in_dout(pe_w_out_w_in_dout[14]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[14]),
      .w_in_read(pe_w_out_w_in_read[14]),
      .w_out_din(pe_w_out_w_in_din[15]),
      .w_out_full_n(pe_w_out_w_in_full_n[15]),
      .w_out_write(pe_w_out_w_in_write[15]));
  // role pe_r2: 2 instance(s)
  pe_r2 u_pe_r2_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[1]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[1]),
      .a_in_read(pe_a_out_a_in_read[1]),
      .a_out_din(pe_a_out_a_in_din[2]),
      .a_out_full_n(pe_a_out_a_in_full_n[2]),
      .a_out_write(pe_a_out_a_in_write[2]),
      .p_out_din(pe_p_out_p_in_din[5]),
      .p_out_full_n(pe_p_out_p_in_full_n[5]),
      .p_out_write(pe_p_out_p_in_write[5]),
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
      .p_out_din(pe_p_out_p_in_din[6]),
      .p_out_full_n(pe_p_out_p_in_full_n[6]),
      .p_out_write(pe_p_out_p_in_write[6]),
      .w_in_dout(pe_w_out_w_in_dout[2]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[2]),
      .w_in_read(pe_w_out_w_in_read[2]),
      .w_out_din(pe_w_out_w_in_din[3]),
      .w_out_full_n(pe_w_out_w_in_full_n[3]),
      .w_out_write(pe_w_out_w_in_write[3]));
  // role pe_r3: 2 instance(s)
  pe_r3 u_pe_r3_1_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[7]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[7]),
      .a_in_read(pe_a_out_a_in_read[7]),
      .p_in_dout(pe_p_out_p_in_dout[7]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[7]),
      .p_in_read(pe_p_out_p_in_read[7]),
      .p_out_din(pe_p_out_p_in_din[11]),
      .p_out_full_n(pe_p_out_p_in_full_n[11]),
      .p_out_write(pe_p_out_p_in_write[11]),
      .w_in_dout(pe_w_out_w_in_dout[7]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[7]),
      .w_in_read(pe_w_out_w_in_read[7]));
  pe_r3 u_pe_r3_2_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[11]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[11]),
      .a_in_read(pe_a_out_a_in_read[11]),
      .p_in_dout(pe_p_out_p_in_dout[11]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[11]),
      .p_in_read(pe_p_out_p_in_read[11]),
      .p_out_din(pe_p_out_p_in_din[15]),
      .p_out_full_n(pe_p_out_p_in_full_n[15]),
      .p_out_write(pe_p_out_p_in_write[15]),
      .w_in_dout(pe_w_out_w_in_dout[11]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[11]),
      .w_in_read(pe_w_out_w_in_read[11]));
  // role pe_r4: 2 instance(s)
  pe_r4 u_pe_r4_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_in_bind_dout[1]),
      .a_in_empty_n(pe_a_in_bind_empty_n[1]),
      .a_in_read(pe_a_in_bind_read[1]),
      .a_out_din(pe_a_out_a_in_din[5]),
      .a_out_full_n(pe_a_out_a_in_full_n[5]),
      .a_out_write(pe_a_out_a_in_write[5]),
      .p_in_dout(pe_p_out_p_in_dout[4]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[4]),
      .p_in_read(pe_p_out_p_in_read[4]),
      .p_out_din(pe_p_out_p_in_din[8]),
      .p_out_full_n(pe_p_out_p_in_full_n[8]),
      .p_out_write(pe_p_out_p_in_write[8]),
      .w_in_dout(pe_w_in_bind_dout[1]),
      .w_in_empty_n(pe_w_in_bind_empty_n[1]),
      .w_in_read(pe_w_in_bind_read[1]),
      .w_out_din(pe_w_out_w_in_din[5]),
      .w_out_full_n(pe_w_out_w_in_full_n[5]),
      .w_out_write(pe_w_out_w_in_write[5]));
  pe_r4 u_pe_r4_2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_in_bind_dout[2]),
      .a_in_empty_n(pe_a_in_bind_empty_n[2]),
      .a_in_read(pe_a_in_bind_read[2]),
      .a_out_din(pe_a_out_a_in_din[9]),
      .a_out_full_n(pe_a_out_a_in_full_n[9]),
      .a_out_write(pe_a_out_a_in_write[9]),
      .p_in_dout(pe_p_out_p_in_dout[8]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[8]),
      .p_in_read(pe_p_out_p_in_read[8]),
      .p_out_din(pe_p_out_p_in_din[12]),
      .p_out_full_n(pe_p_out_p_in_full_n[12]),
      .p_out_write(pe_p_out_p_in_write[12]),
      .w_in_dout(pe_w_in_bind_dout[2]),
      .w_in_empty_n(pe_w_in_bind_empty_n[2]),
      .w_in_read(pe_w_in_bind_read[2]),
      .w_out_din(pe_w_out_w_in_din[9]),
      .w_out_full_n(pe_w_out_w_in_full_n[9]),
      .w_out_write(pe_w_out_w_in_write[9]));
  // role pe_r5: 1 instance(s)
  pe_r5 u_pe_r5_3_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[15]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[15]),
      .a_in_read(pe_a_out_a_in_read[15]),
      .p_in_dout(pe_p_out_p_in_dout[15]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[15]),
      .p_in_read(pe_p_out_p_in_read[15]),
      .p_out_din(lane_z_in_bind_din[3]),
      .p_out_full_n(lane_z_in_bind_full_n[3]),
      .p_out_write(lane_z_in_bind_write[3]),
      .w_in_dout(pe_w_out_w_in_dout[15]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[15]),
      .w_in_read(pe_w_out_w_in_read[15]));
  // role pe_r6: 1 instance(s)
  pe_r6 u_pe_r6_0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[3]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[3]),
      .a_in_read(pe_a_out_a_in_read[3]),
      .p_out_din(pe_p_out_p_in_din[7]),
      .p_out_full_n(pe_p_out_p_in_full_n[7]),
      .p_out_write(pe_p_out_p_in_write[7]),
      .w_in_dout(pe_w_out_w_in_dout[3]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[3]),
      .w_in_read(pe_w_out_w_in_read[3]));
  // role pe_r7: 1 instance(s)
  pe_r7 u_pe_r7_3_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_in_bind_dout[3]),
      .a_in_empty_n(pe_a_in_bind_empty_n[3]),
      .a_in_read(pe_a_in_bind_read[3]),
      .a_out_din(pe_a_out_a_in_din[13]),
      .a_out_full_n(pe_a_out_a_in_full_n[13]),
      .a_out_write(pe_a_out_a_in_write[13]),
      .p_in_dout(pe_p_out_p_in_dout[12]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[12]),
      .p_in_read(pe_p_out_p_in_read[12]),
      .p_out_din(lane_z_in_bind_din[0]),
      .p_out_full_n(lane_z_in_bind_full_n[0]),
      .p_out_write(lane_z_in_bind_write[0]),
      .w_in_dout(pe_w_in_bind_dout[3]),
      .w_in_empty_n(pe_w_in_bind_empty_n[3]),
      .w_in_read(pe_w_in_bind_read[3]),
      .w_out_din(pe_w_out_w_in_din[13]),
      .w_out_full_n(pe_w_out_w_in_full_n[13]),
      .w_out_write(pe_w_out_w_in_write[13]));
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
      .p_out_din(pe_p_out_p_in_din[4]),
      .p_out_full_n(pe_p_out_p_in_full_n[4]),
      .p_out_write(pe_p_out_p_in_write[4]),
      .w_in_dout(pe_w_in_bind_dout[0]),
      .w_in_empty_n(pe_w_in_bind_empty_n[0]),
      .w_in_read(pe_w_in_bind_read[0]),
      .w_out_din(pe_w_out_w_in_din[1]),
      .w_out_full_n(pe_w_out_w_in_full_n[1]),
      .w_out_write(pe_w_out_w_in_write[1]));
  // role etap_r0: 2 instance(s)
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
  // role etap_r1: 1 instance(s)
  etap_r1 u_etap_r1_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_out_din(pe_a_in_bind_din[3]),
      .a_out_full_n(pe_a_in_bind_full_n[3]),
      .a_out_write(pe_a_in_bind_write[3]),
      .e_in_dout(etap_e_out_e_in_dout[3]),
      .e_in_empty_n(etap_e_out_e_in_empty_n[3]),
      .e_in_read(etap_e_out_e_in_read[3]),
      .w_out_din(pe_w_in_bind_din[3]),
      .w_out_full_n(pe_w_in_bind_full_n[3]),
      .w_out_write(pe_w_in_bind_write[3]));
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
  // role req4_r0: 1 instance(s)
  req4_r0 u_req4_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .ins_in_dout(req4_ins_in_bind_dout[0]),
      .ins_in_empty_n(req4_ins_in_bind_empty_n[0]),
      .ins_in_read(req4_ins_in_bind_read[0]),
      .launch_dout(req4_launch_bind_dout[0]),
      .launch_empty_n(req4_launch_bind_empty_n[0]),
      .launch_read(req4_launch_bind_read[0]),
      .op_out_din(head4_op_in_bind_din[0]),
      .op_out_full_n(head4_op_in_bind_full_n[0]),
      .op_out_write(head4_op_in_bind_write[0]),
      .rd_cmd_din(req4_rd_cmd_bind_din[0]),
      .rd_cmd_full_n(req4_rd_cmd_bind_full_n[0]),
      .rd_cmd_write(req4_rd_cmd_bind_write[0]),
      .tag_out_din(deal4_tag_in_bind_din[0]),
      .tag_out_full_n(deal4_tag_in_bind_full_n[0]),
      .tag_out_write(deal4_tag_in_bind_write[0]),
      .y_out_din(wreq4_y_cmd_bind_din[0]),
      .y_out_full_n(wreq4_y_cmd_bind_full_n[0]),
      .y_out_write(wreq4_y_cmd_bind_write[0]));
  // role deal4_r0: 1 instance(s)
  deal4_r0 u_deal4_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_out_din(head4_a_in_bind_din[0]),
      .a_out_full_n(head4_a_in_bind_full_n[0]),
      .a_out_write(head4_a_in_bind_write[0]),
      .b_out_din(head4_b_in_bind_din[0]),
      .b_out_full_n(head4_b_in_bind_full_n[0]),
      .b_out_write(head4_b_in_bind_write[0]),
      .ins_out_din(req4_ins_in_bind_din[0]),
      .ins_out_full_n(req4_ins_in_bind_full_n[0]),
      .ins_out_write(req4_ins_in_bind_write[0]),
      .rd_data_dout(deal4_rd_data_bind_dout[0]),
      .rd_data_empty_n(deal4_rd_data_bind_empty_n[0]),
      .rd_data_read(deal4_rd_data_bind_read[0]),
      .tag_in_dout(deal4_tag_in_bind_dout[0]),
      .tag_in_empty_n(deal4_tag_in_bind_empty_n[0]),
      .tag_in_read(deal4_tag_in_bind_read[0]),
      .w_out_din(head4_w_in_bind_din[0]),
      .w_out_full_n(head4_w_in_bind_full_n[0]),
      .w_out_write(head4_w_in_bind_write[0]));
  // role head4_r0: 1 instance(s)
  head4_r0 u_head4_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(head4_a_in_bind_dout[0]),
      .a_in_empty_n(head4_a_in_bind_empty_n[0]),
      .a_in_read(head4_a_in_bind_read[0]),
      .b_in_dout(head4_b_in_bind_dout[0]),
      .b_in_empty_n(head4_b_in_bind_empty_n[0]),
      .b_in_read(head4_b_in_bind_read[0]),
      .credit_dout(head4_credit_bind_dout[0]),
      .credit_empty_n(head4_credit_bind_empty_n[0]),
      .credit_read(head4_credit_bind_read[0]),
      .e_out_din(etap_e_in_bind_din[0]),
      .e_out_full_n(etap_e_in_bind_full_n[0]),
      .e_out_write(etap_e_in_bind_write[0]),
      .op_in_dout(head4_op_in_bind_dout[0]),
      .op_in_empty_n(head4_op_in_bind_empty_n[0]),
      .op_in_read(head4_op_in_bind_read[0]),
      .u_out_din(uq_u_in_bind_din[0]),
      .u_out_full_n(uq_u_in_bind_full_n[0]),
      .u_out_write(uq_u_in_bind_write[0]),
      .w_in_dout(head4_w_in_bind_dout[0]),
      .w_in_empty_n(head4_w_in_bind_empty_n[0]),
      .w_in_read(head4_w_in_bind_read[0]));
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
  // role tap_r0: 2 instance(s)
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
  // role tap_r1: 1 instance(s)
  tap_r1 u_tap_r1_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_out_din(lane_c_in_bind_din[3]),
      .c_out_full_n(lane_c_in_bind_full_n[3]),
      .c_out_write(lane_c_in_bind_write[3]),
      .u_in_dout(tap_u_out_u_in_dout[3]),
      .u_in_empty_n(tap_u_out_u_in_empty_n[3]),
      .u_in_read(tap_u_out_u_in_read[3]));
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
  // coordinate axis 0: 4 constant source(s)
  wire [31:0] lane_pid0_dout [0:3];
  wire lane_pid0_empty_n [0:3];
  wire lane_pid0_read [0:3];
  spmw_const #(.DW(32), .VAL(0)) u_lane_pid0_0 (.dout(lane_pid0_dout[0]), .empty_n(lane_pid0_empty_n[0]), .read(lane_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_lane_pid0_1 (.dout(lane_pid0_dout[1]), .empty_n(lane_pid0_empty_n[1]), .read(lane_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_lane_pid0_2 (.dout(lane_pid0_dout[2]), .empty_n(lane_pid0_empty_n[2]), .read(lane_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_lane_pid0_3 (.dout(lane_pid0_dout[3]), .empty_n(lane_pid0_empty_n[3]), .read(lane_pid0_read[3]));
  // role lane_r0: 4 instance(s)
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
  // coordinate axis 0: 4 constant source(s)
  wire [31:0] ctap_pid0_dout [0:3];
  wire ctap_pid0_empty_n [0:3];
  wire ctap_pid0_read [0:3];
  spmw_const #(.DW(32), .VAL(0)) u_ctap_pid0_0 (.dout(ctap_pid0_dout[0]), .empty_n(ctap_pid0_empty_n[0]), .read(ctap_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_ctap_pid0_1 (.dout(ctap_pid0_dout[1]), .empty_n(ctap_pid0_empty_n[1]), .read(ctap_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_ctap_pid0_2 (.dout(ctap_pid0_dout[2]), .empty_n(ctap_pid0_empty_n[2]), .read(ctap_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_ctap_pid0_3 (.dout(ctap_pid0_dout[3]), .empty_n(ctap_pid0_empty_n[3]), .read(ctap_pid0_read[3]));
  // role ctap_r0: 2 instance(s)
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
  // role ctap_r1: 1 instance(s)
  ctap_r1 u_ctap_r1_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .r_in_dout(ctap_r_out_r_in_dout[3]),
      .r_in_empty_n(ctap_r_out_r_in_empty_n[3]),
      .r_in_read(ctap_r_out_r_in_read[3]),
      .r_out_din(pack4_row_in_bind_din[0]),
      .r_out_full_n(pack4_row_in_bind_full_n[0]),
      .r_out_write(pack4_row_in_bind_write[0]),
      .y_in_dout(ctap_y_in_bind_dout[3]),
      .y_in_empty_n(ctap_y_in_bind_empty_n[3]),
      .y_in_read(ctap_y_in_bind_read[3]),
      ._pid0_dout(ctap_pid0_dout[3]),
      ._pid0_empty_n(ctap_pid0_empty_n[3]),
      ._pid0_read(ctap_pid0_read[3]));
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
  // role wreq4_r0: 1 instance(s)
  wreq4_r0 u_wreq4_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .done_din(wreq4_done_bind_din[0]),
      .done_full_n(wreq4_done_bind_full_n[0]),
      .done_write(wreq4_done_bind_write[0]),
      .wr_ack_dout(wreq4_wr_ack_bind_dout[0]),
      .wr_ack_empty_n(wreq4_wr_ack_bind_empty_n[0]),
      .wr_ack_read(wreq4_wr_ack_bind_read[0]),
      .wr_cmd_din(wreq4_wr_cmd_bind_din[0]),
      .wr_cmd_full_n(wreq4_wr_cmd_bind_full_n[0]),
      .wr_cmd_write(wreq4_wr_cmd_bind_write[0]),
      .y_cmd_dout(wreq4_y_cmd_bind_dout[0]),
      .y_cmd_empty_n(wreq4_y_cmd_bind_empty_n[0]),
      .y_cmd_read(wreq4_y_cmd_bind_read[0]));
  // role pack4_r0: 1 instance(s)
  pack4_r0 u_pack4_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .credit_din(head4_credit_bind_din[0]),
      .credit_full_n(head4_credit_bind_full_n[0]),
      .credit_write(head4_credit_bind_write[0]),
      .row_in_dout(pack4_row_in_bind_dout[0]),
      .row_in_empty_n(pack4_row_in_bind_empty_n[0]),
      .row_in_read(pack4_row_in_bind_read[0]),
      .wr_data_din(pack4_wr_data_bind_din[0]),
      .wr_data_full_n(pack4_wr_data_bind_full_n[0]),
      .wr_data_write(pack4_wr_data_bind_write[0]));
endmodule
