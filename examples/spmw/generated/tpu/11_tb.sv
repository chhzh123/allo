`timescale 1ns/1ps

module tb;
  reg clk = 0, rst_n = 0;
  always #5 clk = ~clk;
  integer errors = 0;
  integer produced = 0;
  localparam integer TOTAL = 4;
  wire [7:0] mac_a_in_bind_dout [0:3];
  wire mac_a_in_bind_empty_n [0:3];
  wire mac_a_in_bind_read [0:3];
  reg [7:0] mac_a_in_bind_src0 [0:5];
  integer mac_a_in_bind_p0 = 0;
  initial begin
    mac_a_in_bind_src0[0] = 8'h02;
    mac_a_in_bind_src0[1] = 8'h00;
    mac_a_in_bind_src0[2] = 8'h00;
    mac_a_in_bind_src0[3] = 8'h01;
    mac_a_in_bind_src0[4] = 8'h01;
    mac_a_in_bind_src0[5] = 8'h00;
  end
  assign mac_a_in_bind_dout[0] = mac_a_in_bind_src0[mac_a_in_bind_p0 < 6 ? mac_a_in_bind_p0 : 5];
  assign mac_a_in_bind_empty_n[0] = (mac_a_in_bind_p0 < 6);
  always @(posedge clk) if (rst_n && mac_a_in_bind_read[0] && mac_a_in_bind_empty_n[0]) mac_a_in_bind_p0 <= mac_a_in_bind_p0 + 1;
  reg [7:0] mac_a_in_bind_src1 [0:5];
  integer mac_a_in_bind_p1 = 0;
  initial begin
    mac_a_in_bind_src1[0] = 8'h01;
    mac_a_in_bind_src1[1] = 8'h00;
    mac_a_in_bind_src1[2] = 8'h02;
    mac_a_in_bind_src1[3] = 8'h01;
    mac_a_in_bind_src1[4] = 8'h01;
    mac_a_in_bind_src1[5] = 8'h02;
  end
  assign mac_a_in_bind_dout[1] = mac_a_in_bind_src1[mac_a_in_bind_p1 < 6 ? mac_a_in_bind_p1 : 5];
  assign mac_a_in_bind_empty_n[1] = (mac_a_in_bind_p1 < 6);
  always @(posedge clk) if (rst_n && mac_a_in_bind_read[1] && mac_a_in_bind_empty_n[1]) mac_a_in_bind_p1 <= mac_a_in_bind_p1 + 1;
  reg [7:0] mac_a_in_bind_src2 [0:5];
  integer mac_a_in_bind_p2 = 0;
  initial begin
    mac_a_in_bind_src2[0] = 8'h01;
    mac_a_in_bind_src2[1] = 8'h00;
    mac_a_in_bind_src2[2] = 8'h01;
    mac_a_in_bind_src2[3] = 8'h02;
    mac_a_in_bind_src2[4] = 8'h01;
    mac_a_in_bind_src2[5] = 8'h02;
  end
  assign mac_a_in_bind_dout[2] = mac_a_in_bind_src2[mac_a_in_bind_p2 < 6 ? mac_a_in_bind_p2 : 5];
  assign mac_a_in_bind_empty_n[2] = (mac_a_in_bind_p2 < 6);
  always @(posedge clk) if (rst_n && mac_a_in_bind_read[2] && mac_a_in_bind_empty_n[2]) mac_a_in_bind_p2 <= mac_a_in_bind_p2 + 1;
  reg [7:0] mac_a_in_bind_src3 [0:5];
  integer mac_a_in_bind_p3 = 0;
  initial begin
    mac_a_in_bind_src3[0] = 8'h00;
    mac_a_in_bind_src3[1] = 8'h00;
    mac_a_in_bind_src3[2] = 8'h02;
    mac_a_in_bind_src3[3] = 8'h02;
    mac_a_in_bind_src3[4] = 8'h02;
    mac_a_in_bind_src3[5] = 8'h00;
  end
  assign mac_a_in_bind_dout[3] = mac_a_in_bind_src3[mac_a_in_bind_p3 < 6 ? mac_a_in_bind_p3 : 5];
  assign mac_a_in_bind_empty_n[3] = (mac_a_in_bind_p3 < 6);
  always @(posedge clk) if (rst_n && mac_a_in_bind_read[3] && mac_a_in_bind_empty_n[3]) mac_a_in_bind_p3 <= mac_a_in_bind_p3 + 1;
  wire [7:0] mac_w_mem_dout [0:15];
  wire mac_w_mem_empty_n [0:15];
  wire mac_w_mem_read [0:15];
  reg [7:0] mac_w_mem_src0 [0:0];
  integer mac_w_mem_p0 = 0;
  initial begin
    mac_w_mem_src0[0] = 8'h01;
  end
  assign mac_w_mem_dout[0] = mac_w_mem_src0[mac_w_mem_p0 < 1 ? mac_w_mem_p0 : 0];
  assign mac_w_mem_empty_n[0] = (mac_w_mem_p0 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[0] && mac_w_mem_empty_n[0]) mac_w_mem_p0 <= mac_w_mem_p0 + 1;
  reg [7:0] mac_w_mem_src1 [0:0];
  integer mac_w_mem_p1 = 0;
  initial begin
    mac_w_mem_src1[0] = 8'h02;
  end
  assign mac_w_mem_dout[1] = mac_w_mem_src1[mac_w_mem_p1 < 1 ? mac_w_mem_p1 : 0];
  assign mac_w_mem_empty_n[1] = (mac_w_mem_p1 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[1] && mac_w_mem_empty_n[1]) mac_w_mem_p1 <= mac_w_mem_p1 + 1;
  reg [7:0] mac_w_mem_src2 [0:0];
  integer mac_w_mem_p2 = 0;
  initial begin
    mac_w_mem_src2[0] = 8'h01;
  end
  assign mac_w_mem_dout[2] = mac_w_mem_src2[mac_w_mem_p2 < 1 ? mac_w_mem_p2 : 0];
  assign mac_w_mem_empty_n[2] = (mac_w_mem_p2 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[2] && mac_w_mem_empty_n[2]) mac_w_mem_p2 <= mac_w_mem_p2 + 1;
  reg [7:0] mac_w_mem_src3 [0:0];
  integer mac_w_mem_p3 = 0;
  initial begin
    mac_w_mem_src3[0] = 8'h00;
  end
  assign mac_w_mem_dout[3] = mac_w_mem_src3[mac_w_mem_p3 < 1 ? mac_w_mem_p3 : 0];
  assign mac_w_mem_empty_n[3] = (mac_w_mem_p3 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[3] && mac_w_mem_empty_n[3]) mac_w_mem_p3 <= mac_w_mem_p3 + 1;
  reg [7:0] mac_w_mem_src4 [0:0];
  integer mac_w_mem_p4 = 0;
  initial begin
    mac_w_mem_src4[0] = 8'h02;
  end
  assign mac_w_mem_dout[4] = mac_w_mem_src4[mac_w_mem_p4 < 1 ? mac_w_mem_p4 : 0];
  assign mac_w_mem_empty_n[4] = (mac_w_mem_p4 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[4] && mac_w_mem_empty_n[4]) mac_w_mem_p4 <= mac_w_mem_p4 + 1;
  reg [7:0] mac_w_mem_src5 [0:0];
  integer mac_w_mem_p5 = 0;
  initial begin
    mac_w_mem_src5[0] = 8'h02;
  end
  assign mac_w_mem_dout[5] = mac_w_mem_src5[mac_w_mem_p5 < 1 ? mac_w_mem_p5 : 0];
  assign mac_w_mem_empty_n[5] = (mac_w_mem_p5 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[5] && mac_w_mem_empty_n[5]) mac_w_mem_p5 <= mac_w_mem_p5 + 1;
  reg [7:0] mac_w_mem_src6 [0:0];
  integer mac_w_mem_p6 = 0;
  initial begin
    mac_w_mem_src6[0] = 8'h02;
  end
  assign mac_w_mem_dout[6] = mac_w_mem_src6[mac_w_mem_p6 < 1 ? mac_w_mem_p6 : 0];
  assign mac_w_mem_empty_n[6] = (mac_w_mem_p6 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[6] && mac_w_mem_empty_n[6]) mac_w_mem_p6 <= mac_w_mem_p6 + 1;
  reg [7:0] mac_w_mem_src7 [0:0];
  integer mac_w_mem_p7 = 0;
  initial begin
    mac_w_mem_src7[0] = 8'h00;
  end
  assign mac_w_mem_dout[7] = mac_w_mem_src7[mac_w_mem_p7 < 1 ? mac_w_mem_p7 : 0];
  assign mac_w_mem_empty_n[7] = (mac_w_mem_p7 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[7] && mac_w_mem_empty_n[7]) mac_w_mem_p7 <= mac_w_mem_p7 + 1;
  reg [7:0] mac_w_mem_src8 [0:0];
  integer mac_w_mem_p8 = 0;
  initial begin
    mac_w_mem_src8[0] = 8'h00;
  end
  assign mac_w_mem_dout[8] = mac_w_mem_src8[mac_w_mem_p8 < 1 ? mac_w_mem_p8 : 0];
  assign mac_w_mem_empty_n[8] = (mac_w_mem_p8 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[8] && mac_w_mem_empty_n[8]) mac_w_mem_p8 <= mac_w_mem_p8 + 1;
  reg [7:0] mac_w_mem_src9 [0:0];
  integer mac_w_mem_p9 = 0;
  initial begin
    mac_w_mem_src9[0] = 8'h02;
  end
  assign mac_w_mem_dout[9] = mac_w_mem_src9[mac_w_mem_p9 < 1 ? mac_w_mem_p9 : 0];
  assign mac_w_mem_empty_n[9] = (mac_w_mem_p9 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[9] && mac_w_mem_empty_n[9]) mac_w_mem_p9 <= mac_w_mem_p9 + 1;
  reg [7:0] mac_w_mem_src10 [0:0];
  integer mac_w_mem_p10 = 0;
  initial begin
    mac_w_mem_src10[0] = 8'h00;
  end
  assign mac_w_mem_dout[10] = mac_w_mem_src10[mac_w_mem_p10 < 1 ? mac_w_mem_p10 : 0];
  assign mac_w_mem_empty_n[10] = (mac_w_mem_p10 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[10] && mac_w_mem_empty_n[10]) mac_w_mem_p10 <= mac_w_mem_p10 + 1;
  reg [7:0] mac_w_mem_src11 [0:0];
  integer mac_w_mem_p11 = 0;
  initial begin
    mac_w_mem_src11[0] = 8'h01;
  end
  assign mac_w_mem_dout[11] = mac_w_mem_src11[mac_w_mem_p11 < 1 ? mac_w_mem_p11 : 0];
  assign mac_w_mem_empty_n[11] = (mac_w_mem_p11 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[11] && mac_w_mem_empty_n[11]) mac_w_mem_p11 <= mac_w_mem_p11 + 1;
  reg [7:0] mac_w_mem_src12 [0:0];
  integer mac_w_mem_p12 = 0;
  initial begin
    mac_w_mem_src12[0] = 8'h00;
  end
  assign mac_w_mem_dout[12] = mac_w_mem_src12[mac_w_mem_p12 < 1 ? mac_w_mem_p12 : 0];
  assign mac_w_mem_empty_n[12] = (mac_w_mem_p12 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[12] && mac_w_mem_empty_n[12]) mac_w_mem_p12 <= mac_w_mem_p12 + 1;
  reg [7:0] mac_w_mem_src13 [0:0];
  integer mac_w_mem_p13 = 0;
  initial begin
    mac_w_mem_src13[0] = 8'h00;
  end
  assign mac_w_mem_dout[13] = mac_w_mem_src13[mac_w_mem_p13 < 1 ? mac_w_mem_p13 : 0];
  assign mac_w_mem_empty_n[13] = (mac_w_mem_p13 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[13] && mac_w_mem_empty_n[13]) mac_w_mem_p13 <= mac_w_mem_p13 + 1;
  reg [7:0] mac_w_mem_src14 [0:0];
  integer mac_w_mem_p14 = 0;
  initial begin
    mac_w_mem_src14[0] = 8'h01;
  end
  assign mac_w_mem_dout[14] = mac_w_mem_src14[mac_w_mem_p14 < 1 ? mac_w_mem_p14 : 0];
  assign mac_w_mem_empty_n[14] = (mac_w_mem_p14 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[14] && mac_w_mem_empty_n[14]) mac_w_mem_p14 <= mac_w_mem_p14 + 1;
  reg [7:0] mac_w_mem_src15 [0:0];
  integer mac_w_mem_p15 = 0;
  initial begin
    mac_w_mem_src15[0] = 8'h01;
  end
  assign mac_w_mem_dout[15] = mac_w_mem_src15[mac_w_mem_p15 < 1 ? mac_w_mem_p15 : 0];
  assign mac_w_mem_empty_n[15] = (mac_w_mem_p15 < 1);
  always @(posedge clk) if (rst_n && mac_w_mem_read[15] && mac_w_mem_empty_n[15]) mac_w_mem_p15 <= mac_w_mem_p15 + 1;
  wire [7:0] act_y_out_bind_din [0:3];
  wire act_y_out_bind_write [0:3];
  wire act_y_out_bind_full_n [0:3];
  reg [7:0] act_y_out_bind_exp0 [0:5];
  integer act_y_out_bind_q0 = 0;
  initial begin
    act_y_out_bind_exp0[0] = 8'h00;
    act_y_out_bind_exp0[1] = 8'h00;
    act_y_out_bind_exp0[2] = 8'h00;
    act_y_out_bind_exp0[3] = 8'h00;
    act_y_out_bind_exp0[4] = 8'h00;
    act_y_out_bind_exp0[5] = 8'h00;
  end
  assign act_y_out_bind_full_n[0] = 1'b1;
  always @(posedge clk) if (rst_n && act_y_out_bind_write[0]) begin
    if (act_y_out_bind_q0 < 6) begin
      if (act_y_out_bind_din[0] !== act_y_out_bind_exp0[act_y_out_bind_q0]) begin
        errors = errors + 1;
        $display("MISMATCH act_y_out_bind[%0d] step %0d: got %h want %h",
                 0, act_y_out_bind_q0, act_y_out_bind_din[0], act_y_out_bind_exp0[act_y_out_bind_q0]);
      end
      act_y_out_bind_q0 <= act_y_out_bind_q0 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on act_y_out_bind[%0d]", 0);
    end
  end
  reg [7:0] act_y_out_bind_exp1 [0:5];
  integer act_y_out_bind_q1 = 0;
  initial begin
    act_y_out_bind_exp1[0] = 8'h00;
    act_y_out_bind_exp1[1] = 8'h00;
    act_y_out_bind_exp1[2] = 8'h00;
    act_y_out_bind_exp1[3] = 8'h00;
    act_y_out_bind_exp1[4] = 8'h00;
    act_y_out_bind_exp1[5] = 8'h00;
  end
  assign act_y_out_bind_full_n[1] = 1'b1;
  always @(posedge clk) if (rst_n && act_y_out_bind_write[1]) begin
    if (act_y_out_bind_q1 < 6) begin
      if (act_y_out_bind_din[1] !== act_y_out_bind_exp1[act_y_out_bind_q1]) begin
        errors = errors + 1;
        $display("MISMATCH act_y_out_bind[%0d] step %0d: got %h want %h",
                 1, act_y_out_bind_q1, act_y_out_bind_din[1], act_y_out_bind_exp1[act_y_out_bind_q1]);
      end
      act_y_out_bind_q1 <= act_y_out_bind_q1 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on act_y_out_bind[%0d]", 1);
    end
  end
  reg [7:0] act_y_out_bind_exp2 [0:5];
  integer act_y_out_bind_q2 = 0;
  initial begin
    act_y_out_bind_exp2[0] = 8'h00;
    act_y_out_bind_exp2[1] = 8'h00;
    act_y_out_bind_exp2[2] = 8'h00;
    act_y_out_bind_exp2[3] = 8'h00;
    act_y_out_bind_exp2[4] = 8'h00;
    act_y_out_bind_exp2[5] = 8'h00;
  end
  assign act_y_out_bind_full_n[2] = 1'b1;
  always @(posedge clk) if (rst_n && act_y_out_bind_write[2]) begin
    if (act_y_out_bind_q2 < 6) begin
      if (act_y_out_bind_din[2] !== act_y_out_bind_exp2[act_y_out_bind_q2]) begin
        errors = errors + 1;
        $display("MISMATCH act_y_out_bind[%0d] step %0d: got %h want %h",
                 2, act_y_out_bind_q2, act_y_out_bind_din[2], act_y_out_bind_exp2[act_y_out_bind_q2]);
      end
      act_y_out_bind_q2 <= act_y_out_bind_q2 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on act_y_out_bind[%0d]", 2);
    end
  end
  reg [7:0] act_y_out_bind_exp3 [0:5];
  integer act_y_out_bind_q3 = 0;
  initial begin
    act_y_out_bind_exp3[0] = 8'h00;
    act_y_out_bind_exp3[1] = 8'h00;
    act_y_out_bind_exp3[2] = 8'h00;
    act_y_out_bind_exp3[3] = 8'h00;
    act_y_out_bind_exp3[4] = 8'h00;
    act_y_out_bind_exp3[5] = 8'h00;
  end
  assign act_y_out_bind_full_n[3] = 1'b1;
  always @(posedge clk) if (rst_n && act_y_out_bind_write[3]) begin
    if (act_y_out_bind_q3 < 6) begin
      if (act_y_out_bind_din[3] !== act_y_out_bind_exp3[act_y_out_bind_q3]) begin
        errors = errors + 1;
        $display("MISMATCH act_y_out_bind[%0d] step %0d: got %h want %h",
                 3, act_y_out_bind_q3, act_y_out_bind_din[3], act_y_out_bind_exp3[act_y_out_bind_q3]);
      end
      act_y_out_bind_q3 <= act_y_out_bind_q3 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on act_y_out_bind[%0d]", 3);
    end
  end
  spmw_top dut (.ap_clk(clk), .ap_rst_n(rst_n), .mac_a_in_bind_dout(mac_a_in_bind_dout), .mac_a_in_bind_empty_n(mac_a_in_bind_empty_n), .mac_a_in_bind_read(mac_a_in_bind_read), .mac_w_mem_dout(mac_w_mem_dout), .mac_w_mem_empty_n(mac_w_mem_empty_n), .mac_w_mem_read(mac_w_mem_read), .act_y_out_bind_din(act_y_out_bind_din), .act_y_out_bind_write(act_y_out_bind_write), .act_y_out_bind_full_n(act_y_out_bind_full_n));
  initial begin
    repeat (4) @(posedge clk);
    rst_n = 1;
    for (integer c = 0; c < 200000; c = c + 1) begin
      @(posedge clk);
      if (produced == TOTAL) begin
        $display("SPMW COSIM %s (%0d/%0d tokens, %0d errors)",
                 errors == 0 ? "PASS" : "FAIL", produced, TOTAL, errors);
        $finish;
      end
    end
    $display("SPMW COSIM TIMEOUT (%0d/%0d tokens, %0d errors)",
             produced, TOTAL, errors);
    $finish;
  end
endmodule
