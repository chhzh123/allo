`timescale 1ns/1ps

module tb;
  reg clk = 0, rst_n = 0;
  always #5 clk = ~clk;
  integer errors = 0;
  integer produced = 0;
  integer first = -1;
  localparam integer TOTAL = 24;
  wire [7:0] tiled_mac_a_in_bind_dout [0:3];
  wire tiled_mac_a_in_bind_empty_n [0:3];
  wire tiled_mac_a_in_bind_read [0:3];
  reg [7:0] tiled_mac_a_in_bind_src0 [0:11];
  integer tiled_mac_a_in_bind_p0 = 0;
  initial begin
    tiled_mac_a_in_bind_src0[0] = 8'h02;
    tiled_mac_a_in_bind_src0[1] = 8'h00;
    tiled_mac_a_in_bind_src0[2] = 8'h00;
    tiled_mac_a_in_bind_src0[3] = 8'h01;
    tiled_mac_a_in_bind_src0[4] = 8'h01;
    tiled_mac_a_in_bind_src0[5] = 8'h00;
    tiled_mac_a_in_bind_src0[6] = 8'h01;
    tiled_mac_a_in_bind_src0[7] = 8'h02;
    tiled_mac_a_in_bind_src0[8] = 8'h00;
    tiled_mac_a_in_bind_src0[9] = 8'h00;
    tiled_mac_a_in_bind_src0[10] = 8'h01;
    tiled_mac_a_in_bind_src0[11] = 8'h00;
  end
  assign tiled_mac_a_in_bind_dout[0] = tiled_mac_a_in_bind_src0[tiled_mac_a_in_bind_p0 < 12 ? tiled_mac_a_in_bind_p0 : 11];
  assign tiled_mac_a_in_bind_empty_n[0] = (tiled_mac_a_in_bind_p0 < 12);
  always @(posedge clk) if (rst_n && tiled_mac_a_in_bind_read[0] && tiled_mac_a_in_bind_empty_n[0]) tiled_mac_a_in_bind_p0 <= tiled_mac_a_in_bind_p0 + 1;
  reg [7:0] tiled_mac_a_in_bind_src1 [0:11];
  integer tiled_mac_a_in_bind_p1 = 0;
  initial begin
    tiled_mac_a_in_bind_src1[0] = 8'h01;
    tiled_mac_a_in_bind_src1[1] = 8'h00;
    tiled_mac_a_in_bind_src1[2] = 8'h02;
    tiled_mac_a_in_bind_src1[3] = 8'h01;
    tiled_mac_a_in_bind_src1[4] = 8'h01;
    tiled_mac_a_in_bind_src1[5] = 8'h02;
    tiled_mac_a_in_bind_src1[6] = 8'h02;
    tiled_mac_a_in_bind_src1[7] = 8'h02;
    tiled_mac_a_in_bind_src1[8] = 8'h02;
    tiled_mac_a_in_bind_src1[9] = 8'h00;
    tiled_mac_a_in_bind_src1[10] = 8'h00;
    tiled_mac_a_in_bind_src1[11] = 8'h02;
  end
  assign tiled_mac_a_in_bind_dout[1] = tiled_mac_a_in_bind_src1[tiled_mac_a_in_bind_p1 < 12 ? tiled_mac_a_in_bind_p1 : 11];
  assign tiled_mac_a_in_bind_empty_n[1] = (tiled_mac_a_in_bind_p1 < 12);
  always @(posedge clk) if (rst_n && tiled_mac_a_in_bind_read[1] && tiled_mac_a_in_bind_empty_n[1]) tiled_mac_a_in_bind_p1 <= tiled_mac_a_in_bind_p1 + 1;
  reg [7:0] tiled_mac_a_in_bind_src2 [0:11];
  integer tiled_mac_a_in_bind_p2 = 0;
  initial begin
    tiled_mac_a_in_bind_src2[0] = 8'h01;
    tiled_mac_a_in_bind_src2[1] = 8'h00;
    tiled_mac_a_in_bind_src2[2] = 8'h01;
    tiled_mac_a_in_bind_src2[3] = 8'h02;
    tiled_mac_a_in_bind_src2[4] = 8'h01;
    tiled_mac_a_in_bind_src2[5] = 8'h02;
    tiled_mac_a_in_bind_src2[6] = 8'h01;
    tiled_mac_a_in_bind_src2[7] = 8'h02;
    tiled_mac_a_in_bind_src2[8] = 8'h00;
    tiled_mac_a_in_bind_src2[9] = 8'h01;
    tiled_mac_a_in_bind_src2[10] = 8'h00;
    tiled_mac_a_in_bind_src2[11] = 8'h01;
  end
  assign tiled_mac_a_in_bind_dout[2] = tiled_mac_a_in_bind_src2[tiled_mac_a_in_bind_p2 < 12 ? tiled_mac_a_in_bind_p2 : 11];
  assign tiled_mac_a_in_bind_empty_n[2] = (tiled_mac_a_in_bind_p2 < 12);
  always @(posedge clk) if (rst_n && tiled_mac_a_in_bind_read[2] && tiled_mac_a_in_bind_empty_n[2]) tiled_mac_a_in_bind_p2 <= tiled_mac_a_in_bind_p2 + 1;
  reg [7:0] tiled_mac_a_in_bind_src3 [0:11];
  integer tiled_mac_a_in_bind_p3 = 0;
  initial begin
    tiled_mac_a_in_bind_src3[0] = 8'h00;
    tiled_mac_a_in_bind_src3[1] = 8'h00;
    tiled_mac_a_in_bind_src3[2] = 8'h02;
    tiled_mac_a_in_bind_src3[3] = 8'h02;
    tiled_mac_a_in_bind_src3[4] = 8'h02;
    tiled_mac_a_in_bind_src3[5] = 8'h00;
    tiled_mac_a_in_bind_src3[6] = 8'h00;
    tiled_mac_a_in_bind_src3[7] = 8'h00;
    tiled_mac_a_in_bind_src3[8] = 8'h01;
    tiled_mac_a_in_bind_src3[9] = 8'h01;
    tiled_mac_a_in_bind_src3[10] = 8'h00;
    tiled_mac_a_in_bind_src3[11] = 8'h01;
  end
  assign tiled_mac_a_in_bind_dout[3] = tiled_mac_a_in_bind_src3[tiled_mac_a_in_bind_p3 < 12 ? tiled_mac_a_in_bind_p3 : 11];
  assign tiled_mac_a_in_bind_empty_n[3] = (tiled_mac_a_in_bind_p3 < 12);
  always @(posedge clk) if (rst_n && tiled_mac_a_in_bind_read[3] && tiled_mac_a_in_bind_empty_n[3]) tiled_mac_a_in_bind_p3 <= tiled_mac_a_in_bind_p3 + 1;
  wire [15:0] tiled_mac_w_mem_dout [0:15];
  wire tiled_mac_w_mem_empty_n [0:15];
  wire tiled_mac_w_mem_read [0:15];
  reg [15:0] tiled_mac_w_mem_src0 [0:0];
  integer tiled_mac_w_mem_p0 = 0;
  initial begin
    tiled_mac_w_mem_src0[0] = 16'h0100;
  end
  assign tiled_mac_w_mem_dout[0] = tiled_mac_w_mem_src0[tiled_mac_w_mem_p0 < 1 ? tiled_mac_w_mem_p0 : 0];
  assign tiled_mac_w_mem_empty_n[0] = (tiled_mac_w_mem_p0 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[0] && tiled_mac_w_mem_empty_n[0]) tiled_mac_w_mem_p0 <= tiled_mac_w_mem_p0 + 1;
  reg [15:0] tiled_mac_w_mem_src1 [0:0];
  integer tiled_mac_w_mem_p1 = 0;
  initial begin
    tiled_mac_w_mem_src1[0] = 16'h0102;
  end
  assign tiled_mac_w_mem_dout[1] = tiled_mac_w_mem_src1[tiled_mac_w_mem_p1 < 1 ? tiled_mac_w_mem_p1 : 0];
  assign tiled_mac_w_mem_empty_n[1] = (tiled_mac_w_mem_p1 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[1] && tiled_mac_w_mem_empty_n[1]) tiled_mac_w_mem_p1 <= tiled_mac_w_mem_p1 + 1;
  reg [15:0] tiled_mac_w_mem_src2 [0:0];
  integer tiled_mac_w_mem_p2 = 0;
  initial begin
    tiled_mac_w_mem_src2[0] = 16'h0201;
  end
  assign tiled_mac_w_mem_dout[2] = tiled_mac_w_mem_src2[tiled_mac_w_mem_p2 < 1 ? tiled_mac_w_mem_p2 : 0];
  assign tiled_mac_w_mem_empty_n[2] = (tiled_mac_w_mem_p2 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[2] && tiled_mac_w_mem_empty_n[2]) tiled_mac_w_mem_p2 <= tiled_mac_w_mem_p2 + 1;
  reg [15:0] tiled_mac_w_mem_src3 [0:0];
  integer tiled_mac_w_mem_p3 = 0;
  initial begin
    tiled_mac_w_mem_src3[0] = 16'h0202;
  end
  assign tiled_mac_w_mem_dout[3] = tiled_mac_w_mem_src3[tiled_mac_w_mem_p3 < 1 ? tiled_mac_w_mem_p3 : 0];
  assign tiled_mac_w_mem_empty_n[3] = (tiled_mac_w_mem_p3 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[3] && tiled_mac_w_mem_empty_n[3]) tiled_mac_w_mem_p3 <= tiled_mac_w_mem_p3 + 1;
  reg [15:0] tiled_mac_w_mem_src4 [0:0];
  integer tiled_mac_w_mem_p4 = 0;
  initial begin
    tiled_mac_w_mem_src4[0] = 16'h0201;
  end
  assign tiled_mac_w_mem_dout[4] = tiled_mac_w_mem_src4[tiled_mac_w_mem_p4 < 1 ? tiled_mac_w_mem_p4 : 0];
  assign tiled_mac_w_mem_empty_n[4] = (tiled_mac_w_mem_p4 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[4] && tiled_mac_w_mem_empty_n[4]) tiled_mac_w_mem_p4 <= tiled_mac_w_mem_p4 + 1;
  reg [15:0] tiled_mac_w_mem_src5 [0:0];
  integer tiled_mac_w_mem_p5 = 0;
  initial begin
    tiled_mac_w_mem_src5[0] = 16'h0102;
  end
  assign tiled_mac_w_mem_dout[5] = tiled_mac_w_mem_src5[tiled_mac_w_mem_p5 < 1 ? tiled_mac_w_mem_p5 : 0];
  assign tiled_mac_w_mem_empty_n[5] = (tiled_mac_w_mem_p5 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[5] && tiled_mac_w_mem_empty_n[5]) tiled_mac_w_mem_p5 <= tiled_mac_w_mem_p5 + 1;
  reg [15:0] tiled_mac_w_mem_src6 [0:0];
  integer tiled_mac_w_mem_p6 = 0;
  initial begin
    tiled_mac_w_mem_src6[0] = 16'h0202;
  end
  assign tiled_mac_w_mem_dout[6] = tiled_mac_w_mem_src6[tiled_mac_w_mem_p6 < 1 ? tiled_mac_w_mem_p6 : 0];
  assign tiled_mac_w_mem_empty_n[6] = (tiled_mac_w_mem_p6 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[6] && tiled_mac_w_mem_empty_n[6]) tiled_mac_w_mem_p6 <= tiled_mac_w_mem_p6 + 1;
  reg [15:0] tiled_mac_w_mem_src7 [0:0];
  integer tiled_mac_w_mem_p7 = 0;
  initial begin
    tiled_mac_w_mem_src7[0] = 16'h0102;
  end
  assign tiled_mac_w_mem_dout[7] = tiled_mac_w_mem_src7[tiled_mac_w_mem_p7 < 1 ? tiled_mac_w_mem_p7 : 0];
  assign tiled_mac_w_mem_empty_n[7] = (tiled_mac_w_mem_p7 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[7] && tiled_mac_w_mem_empty_n[7]) tiled_mac_w_mem_p7 <= tiled_mac_w_mem_p7 + 1;
  reg [15:0] tiled_mac_w_mem_src8 [0:0];
  integer tiled_mac_w_mem_p8 = 0;
  initial begin
    tiled_mac_w_mem_src8[0] = 16'h0002;
  end
  assign tiled_mac_w_mem_dout[8] = tiled_mac_w_mem_src8[tiled_mac_w_mem_p8 < 1 ? tiled_mac_w_mem_p8 : 0];
  assign tiled_mac_w_mem_empty_n[8] = (tiled_mac_w_mem_p8 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[8] && tiled_mac_w_mem_empty_n[8]) tiled_mac_w_mem_p8 <= tiled_mac_w_mem_p8 + 1;
  reg [15:0] tiled_mac_w_mem_src9 [0:0];
  integer tiled_mac_w_mem_p9 = 0;
  initial begin
    tiled_mac_w_mem_src9[0] = 16'h0201;
  end
  assign tiled_mac_w_mem_dout[9] = tiled_mac_w_mem_src9[tiled_mac_w_mem_p9 < 1 ? tiled_mac_w_mem_p9 : 0];
  assign tiled_mac_w_mem_empty_n[9] = (tiled_mac_w_mem_p9 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[9] && tiled_mac_w_mem_empty_n[9]) tiled_mac_w_mem_p9 <= tiled_mac_w_mem_p9 + 1;
  reg [15:0] tiled_mac_w_mem_src10 [0:0];
  integer tiled_mac_w_mem_p10 = 0;
  initial begin
    tiled_mac_w_mem_src10[0] = 16'h0102;
  end
  assign tiled_mac_w_mem_dout[10] = tiled_mac_w_mem_src10[tiled_mac_w_mem_p10 < 1 ? tiled_mac_w_mem_p10 : 0];
  assign tiled_mac_w_mem_empty_n[10] = (tiled_mac_w_mem_p10 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[10] && tiled_mac_w_mem_empty_n[10]) tiled_mac_w_mem_p10 <= tiled_mac_w_mem_p10 + 1;
  reg [15:0] tiled_mac_w_mem_src11 [0:0];
  integer tiled_mac_w_mem_p11 = 0;
  initial begin
    tiled_mac_w_mem_src11[0] = 16'h0001;
  end
  assign tiled_mac_w_mem_dout[11] = tiled_mac_w_mem_src11[tiled_mac_w_mem_p11 < 1 ? tiled_mac_w_mem_p11 : 0];
  assign tiled_mac_w_mem_empty_n[11] = (tiled_mac_w_mem_p11 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[11] && tiled_mac_w_mem_empty_n[11]) tiled_mac_w_mem_p11 <= tiled_mac_w_mem_p11 + 1;
  reg [15:0] tiled_mac_w_mem_src12 [0:0];
  integer tiled_mac_w_mem_p12 = 0;
  initial begin
    tiled_mac_w_mem_src12[0] = 16'h0101;
  end
  assign tiled_mac_w_mem_dout[12] = tiled_mac_w_mem_src12[tiled_mac_w_mem_p12 < 1 ? tiled_mac_w_mem_p12 : 0];
  assign tiled_mac_w_mem_empty_n[12] = (tiled_mac_w_mem_p12 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[12] && tiled_mac_w_mem_empty_n[12]) tiled_mac_w_mem_p12 <= tiled_mac_w_mem_p12 + 1;
  reg [15:0] tiled_mac_w_mem_src13 [0:0];
  integer tiled_mac_w_mem_p13 = 0;
  initial begin
    tiled_mac_w_mem_src13[0] = 16'h0202;
  end
  assign tiled_mac_w_mem_dout[13] = tiled_mac_w_mem_src13[tiled_mac_w_mem_p13 < 1 ? tiled_mac_w_mem_p13 : 0];
  assign tiled_mac_w_mem_empty_n[13] = (tiled_mac_w_mem_p13 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[13] && tiled_mac_w_mem_empty_n[13]) tiled_mac_w_mem_p13 <= tiled_mac_w_mem_p13 + 1;
  reg [15:0] tiled_mac_w_mem_src14 [0:0];
  integer tiled_mac_w_mem_p14 = 0;
  initial begin
    tiled_mac_w_mem_src14[0] = 16'h0200;
  end
  assign tiled_mac_w_mem_dout[14] = tiled_mac_w_mem_src14[tiled_mac_w_mem_p14 < 1 ? tiled_mac_w_mem_p14 : 0];
  assign tiled_mac_w_mem_empty_n[14] = (tiled_mac_w_mem_p14 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[14] && tiled_mac_w_mem_empty_n[14]) tiled_mac_w_mem_p14 <= tiled_mac_w_mem_p14 + 1;
  reg [15:0] tiled_mac_w_mem_src15 [0:0];
  integer tiled_mac_w_mem_p15 = 0;
  initial begin
    tiled_mac_w_mem_src15[0] = 16'h0101;
  end
  assign tiled_mac_w_mem_dout[15] = tiled_mac_w_mem_src15[tiled_mac_w_mem_p15 < 1 ? tiled_mac_w_mem_p15 : 0];
  assign tiled_mac_w_mem_empty_n[15] = (tiled_mac_w_mem_p15 < 1);
  always @(posedge clk) if (rst_n && tiled_mac_w_mem_read[15] && tiled_mac_w_mem_empty_n[15]) tiled_mac_w_mem_p15 <= tiled_mac_w_mem_p15 + 1;
  wire [31:0] tiled_vpu_op_in_bind_dout [0:0];
  wire tiled_vpu_op_in_bind_empty_n [0:0];
  wire tiled_vpu_op_in_bind_read [0:0];
  reg [31:0] tiled_vpu_op_in_bind_src0 [0:11];
  integer tiled_vpu_op_in_bind_p0 = 0;
  initial begin
    tiled_vpu_op_in_bind_src0[0] = 32'h03000000;
    tiled_vpu_op_in_bind_src0[1] = 32'h09000000;
    tiled_vpu_op_in_bind_src0[2] = 32'h09000000;
    tiled_vpu_op_in_bind_src0[3] = 32'h02100000;
    tiled_vpu_op_in_bind_src0[4] = 32'h04010000;
    tiled_vpu_op_in_bind_src0[5] = 32'h03200000;
    tiled_vpu_op_in_bind_src0[6] = 32'h06020000;
    tiled_vpu_op_in_bind_src0[7] = 32'h07000004;
    tiled_vpu_op_in_bind_src0[8] = 32'h08000000;
    tiled_vpu_op_in_bind_src0[9] = 32'h00000000;
    tiled_vpu_op_in_bind_src0[10] = 32'h00000000;
    tiled_vpu_op_in_bind_src0[11] = 32'h00000000;
  end
  assign tiled_vpu_op_in_bind_dout[0] = tiled_vpu_op_in_bind_src0[tiled_vpu_op_in_bind_p0 < 12 ? tiled_vpu_op_in_bind_p0 : 11];
  assign tiled_vpu_op_in_bind_empty_n[0] = (tiled_vpu_op_in_bind_p0 < 12);
  always @(posedge clk) if (rst_n && tiled_vpu_op_in_bind_read[0] && tiled_vpu_op_in_bind_empty_n[0]) tiled_vpu_op_in_bind_p0 <= tiled_vpu_op_in_bind_p0 + 1;
  wire [31:0] tiled_vpu_b_mem_dout [0:3];
  wire tiled_vpu_b_mem_empty_n [0:3];
  wire tiled_vpu_b_mem_read [0:3];
  reg [31:0] tiled_vpu_b_mem_src0 [0:0];
  integer tiled_vpu_b_mem_p0 = 0;
  initial begin
    tiled_vpu_b_mem_src0[0] = 32'h00000002;
  end
  assign tiled_vpu_b_mem_dout[0] = tiled_vpu_b_mem_src0[tiled_vpu_b_mem_p0 < 1 ? tiled_vpu_b_mem_p0 : 0];
  assign tiled_vpu_b_mem_empty_n[0] = (tiled_vpu_b_mem_p0 < 1);
  always @(posedge clk) if (rst_n && tiled_vpu_b_mem_read[0] && tiled_vpu_b_mem_empty_n[0]) tiled_vpu_b_mem_p0 <= tiled_vpu_b_mem_p0 + 1;
  reg [31:0] tiled_vpu_b_mem_src1 [0:0];
  integer tiled_vpu_b_mem_p1 = 0;
  initial begin
    tiled_vpu_b_mem_src1[0] = 32'h00000001;
  end
  assign tiled_vpu_b_mem_dout[1] = tiled_vpu_b_mem_src1[tiled_vpu_b_mem_p1 < 1 ? tiled_vpu_b_mem_p1 : 0];
  assign tiled_vpu_b_mem_empty_n[1] = (tiled_vpu_b_mem_p1 < 1);
  always @(posedge clk) if (rst_n && tiled_vpu_b_mem_read[1] && tiled_vpu_b_mem_empty_n[1]) tiled_vpu_b_mem_p1 <= tiled_vpu_b_mem_p1 + 1;
  reg [31:0] tiled_vpu_b_mem_src2 [0:0];
  integer tiled_vpu_b_mem_p2 = 0;
  initial begin
    tiled_vpu_b_mem_src2[0] = 32'h00000000;
  end
  assign tiled_vpu_b_mem_dout[2] = tiled_vpu_b_mem_src2[tiled_vpu_b_mem_p2 < 1 ? tiled_vpu_b_mem_p2 : 0];
  assign tiled_vpu_b_mem_empty_n[2] = (tiled_vpu_b_mem_p2 < 1);
  always @(posedge clk) if (rst_n && tiled_vpu_b_mem_read[2] && tiled_vpu_b_mem_empty_n[2]) tiled_vpu_b_mem_p2 <= tiled_vpu_b_mem_p2 + 1;
  reg [31:0] tiled_vpu_b_mem_src3 [0:0];
  integer tiled_vpu_b_mem_p3 = 0;
  initial begin
    tiled_vpu_b_mem_src3[0] = 32'h00000000;
  end
  assign tiled_vpu_b_mem_dout[3] = tiled_vpu_b_mem_src3[tiled_vpu_b_mem_p3 < 1 ? tiled_vpu_b_mem_p3 : 0];
  assign tiled_vpu_b_mem_empty_n[3] = (tiled_vpu_b_mem_p3 < 1);
  always @(posedge clk) if (rst_n && tiled_vpu_b_mem_read[3] && tiled_vpu_b_mem_empty_n[3]) tiled_vpu_b_mem_p3 <= tiled_vpu_b_mem_p3 + 1;
  wire [31:0] tiled_vpu_y_out_bind_din [0:3];
  wire tiled_vpu_y_out_bind_write [0:3];
  wire tiled_vpu_y_out_bind_full_n [0:3];
  reg [31:0] tiled_vpu_y_out_bind_exp0 [0:5];
  integer tiled_vpu_y_out_bind_q0 = 0;
  initial begin
    tiled_vpu_y_out_bind_exp0[0] = 32'h00000000;
    tiled_vpu_y_out_bind_exp0[1] = 32'h00000000;
    tiled_vpu_y_out_bind_exp0[2] = 32'h00000000;
    tiled_vpu_y_out_bind_exp0[3] = 32'h00000000;
    tiled_vpu_y_out_bind_exp0[4] = 32'h00000000;
    tiled_vpu_y_out_bind_exp0[5] = 32'h00000000;
  end
  assign tiled_vpu_y_out_bind_full_n[0] = 1'b1;
  always @(posedge clk) if (rst_n && tiled_vpu_y_out_bind_write[0]) begin
    if (tiled_vpu_y_out_bind_q0 < 6) begin
      if (tiled_vpu_y_out_bind_din[0] !== tiled_vpu_y_out_bind_exp0[tiled_vpu_y_out_bind_q0]) begin
        errors = errors + 1;
        $display("MISMATCH tiled_vpu_y_out_bind[%0d] step %0d: got %h want %h",
                 0, tiled_vpu_y_out_bind_q0, tiled_vpu_y_out_bind_din[0], tiled_vpu_y_out_bind_exp0[tiled_vpu_y_out_bind_q0]);
      end
      tiled_vpu_y_out_bind_q0 <= tiled_vpu_y_out_bind_q0 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on tiled_vpu_y_out_bind[%0d]", 0);
    end
  end
  reg [31:0] tiled_vpu_y_out_bind_exp1 [0:5];
  integer tiled_vpu_y_out_bind_q1 = 0;
  initial begin
    tiled_vpu_y_out_bind_exp1[0] = 32'h00000000;
    tiled_vpu_y_out_bind_exp1[1] = 32'h00000001;
    tiled_vpu_y_out_bind_exp1[2] = 32'h00000001;
    tiled_vpu_y_out_bind_exp1[3] = 32'h00000001;
    tiled_vpu_y_out_bind_exp1[4] = 32'h00000000;
    tiled_vpu_y_out_bind_exp1[5] = 32'h00000000;
  end
  assign tiled_vpu_y_out_bind_full_n[1] = 1'b1;
  always @(posedge clk) if (rst_n && tiled_vpu_y_out_bind_write[1]) begin
    if (tiled_vpu_y_out_bind_q1 < 6) begin
      if (tiled_vpu_y_out_bind_din[1] !== tiled_vpu_y_out_bind_exp1[tiled_vpu_y_out_bind_q1]) begin
        errors = errors + 1;
        $display("MISMATCH tiled_vpu_y_out_bind[%0d] step %0d: got %h want %h",
                 1, tiled_vpu_y_out_bind_q1, tiled_vpu_y_out_bind_din[1], tiled_vpu_y_out_bind_exp1[tiled_vpu_y_out_bind_q1]);
      end
      tiled_vpu_y_out_bind_q1 <= tiled_vpu_y_out_bind_q1 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on tiled_vpu_y_out_bind[%0d]", 1);
    end
  end
  reg [31:0] tiled_vpu_y_out_bind_exp2 [0:5];
  integer tiled_vpu_y_out_bind_q2 = 0;
  initial begin
    tiled_vpu_y_out_bind_exp2[0] = 32'h00000000;
    tiled_vpu_y_out_bind_exp2[1] = 32'h00000001;
    tiled_vpu_y_out_bind_exp2[2] = 32'h00000000;
    tiled_vpu_y_out_bind_exp2[3] = 32'h00000001;
    tiled_vpu_y_out_bind_exp2[4] = 32'h00000000;
    tiled_vpu_y_out_bind_exp2[5] = 32'h00000000;
  end
  assign tiled_vpu_y_out_bind_full_n[2] = 1'b1;
  always @(posedge clk) if (rst_n && tiled_vpu_y_out_bind_write[2]) begin
    if (tiled_vpu_y_out_bind_q2 < 6) begin
      if (tiled_vpu_y_out_bind_din[2] !== tiled_vpu_y_out_bind_exp2[tiled_vpu_y_out_bind_q2]) begin
        errors = errors + 1;
        $display("MISMATCH tiled_vpu_y_out_bind[%0d] step %0d: got %h want %h",
                 2, tiled_vpu_y_out_bind_q2, tiled_vpu_y_out_bind_din[2], tiled_vpu_y_out_bind_exp2[tiled_vpu_y_out_bind_q2]);
      end
      tiled_vpu_y_out_bind_q2 <= tiled_vpu_y_out_bind_q2 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on tiled_vpu_y_out_bind[%0d]", 2);
    end
  end
  reg [31:0] tiled_vpu_y_out_bind_exp3 [0:5];
  integer tiled_vpu_y_out_bind_q3 = 0;
  initial begin
    tiled_vpu_y_out_bind_exp3[0] = 32'h00000000;
    tiled_vpu_y_out_bind_exp3[1] = 32'h00000000;
    tiled_vpu_y_out_bind_exp3[2] = 32'h00000000;
    tiled_vpu_y_out_bind_exp3[3] = 32'h00000000;
    tiled_vpu_y_out_bind_exp3[4] = 32'h00000000;
    tiled_vpu_y_out_bind_exp3[5] = 32'h00000000;
  end
  assign tiled_vpu_y_out_bind_full_n[3] = 1'b1;
  always @(posedge clk) if (rst_n && tiled_vpu_y_out_bind_write[3]) begin
    if (tiled_vpu_y_out_bind_q3 < 6) begin
      if (tiled_vpu_y_out_bind_din[3] !== tiled_vpu_y_out_bind_exp3[tiled_vpu_y_out_bind_q3]) begin
        errors = errors + 1;
        $display("MISMATCH tiled_vpu_y_out_bind[%0d] step %0d: got %h want %h",
                 3, tiled_vpu_y_out_bind_q3, tiled_vpu_y_out_bind_din[3], tiled_vpu_y_out_bind_exp3[tiled_vpu_y_out_bind_q3]);
      end
      tiled_vpu_y_out_bind_q3 <= tiled_vpu_y_out_bind_q3 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on tiled_vpu_y_out_bind[%0d]", 3);
    end
  end
  spmw_top dut (.ap_clk(clk), .ap_rst_n(rst_n), .tiled_mac_a_in_bind_dout(tiled_mac_a_in_bind_dout), .tiled_mac_a_in_bind_empty_n(tiled_mac_a_in_bind_empty_n), .tiled_mac_a_in_bind_read(tiled_mac_a_in_bind_read), .tiled_mac_w_mem_dout(tiled_mac_w_mem_dout), .tiled_mac_w_mem_empty_n(tiled_mac_w_mem_empty_n), .tiled_mac_w_mem_read(tiled_mac_w_mem_read), .tiled_vpu_op_in_bind_dout(tiled_vpu_op_in_bind_dout), .tiled_vpu_op_in_bind_empty_n(tiled_vpu_op_in_bind_empty_n), .tiled_vpu_op_in_bind_read(tiled_vpu_op_in_bind_read), .tiled_vpu_b_mem_dout(tiled_vpu_b_mem_dout), .tiled_vpu_b_mem_empty_n(tiled_vpu_b_mem_empty_n), .tiled_vpu_b_mem_read(tiled_vpu_b_mem_read), .tiled_vpu_y_out_bind_din(tiled_vpu_y_out_bind_din), .tiled_vpu_y_out_bind_write(tiled_vpu_y_out_bind_write), .tiled_vpu_y_out_bind_full_n(tiled_vpu_y_out_bind_full_n));
  initial begin
    repeat (4) @(posedge clk);
    rst_n = 1;
    for (integer c = 0; c < 200000; c = c + 1) begin
      @(posedge clk);
      if (produced > 0 && first < 0) first = c;
      if (produced == TOTAL) begin
        $display("SPMW COSIM %s (%0d/%0d tokens, %0d errors)",
                 errors == 0 ? "PASS" : "FAIL", produced, TOTAL, errors);
        $display("SPMW CYCLES total=%0d first_out=%0d",
                 c + 1, first + 1);
        $finish;
      end
    end
    $display("SPMW COSIM TIMEOUT (%0d/%0d tokens, %0d errors)",
             produced, TOTAL, errors);
    $display("SPMW CYCLES total=-1 first_out=%0d", first + 1);
    $finish;
  end
endmodule
