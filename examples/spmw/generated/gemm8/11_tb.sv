`timescale 1ns/1ps

module tb;
  reg clk = 0, rst_n = 0;
  always #5 clk = ~clk;
  integer errors = 0;
  integer produced = 0;
  integer first = -1;
  localparam integer TOTAL = 16;
  wire [7:0] pe_west_bind_dout [0:3];
  wire pe_west_bind_empty_n [0:3];
  wire pe_west_bind_read [0:3];
  reg [7:0] pe_west_bind_src0 [0:3];
  integer pe_west_bind_p0 = 0;
  initial begin
    pe_west_bind_src0[0] = 8'h02;
    pe_west_bind_src0[1] = 8'h01;
    pe_west_bind_src0[2] = 8'h01;
    pe_west_bind_src0[3] = 8'h00;
  end
  assign pe_west_bind_dout[0] = pe_west_bind_src0[pe_west_bind_p0 < 4 ? pe_west_bind_p0 : 3];
  assign pe_west_bind_empty_n[0] = (pe_west_bind_p0 < 4);
  always @(posedge clk) if (rst_n && pe_west_bind_read[0] && pe_west_bind_empty_n[0]) pe_west_bind_p0 <= pe_west_bind_p0 + 1;
  reg [7:0] pe_west_bind_src1 [0:3];
  integer pe_west_bind_p1 = 0;
  initial begin
    pe_west_bind_src1[0] = 8'h00;
    pe_west_bind_src1[1] = 8'h00;
    pe_west_bind_src1[2] = 8'h00;
    pe_west_bind_src1[3] = 8'h00;
  end
  assign pe_west_bind_dout[1] = pe_west_bind_src1[pe_west_bind_p1 < 4 ? pe_west_bind_p1 : 3];
  assign pe_west_bind_empty_n[1] = (pe_west_bind_p1 < 4);
  always @(posedge clk) if (rst_n && pe_west_bind_read[1] && pe_west_bind_empty_n[1]) pe_west_bind_p1 <= pe_west_bind_p1 + 1;
  reg [7:0] pe_west_bind_src2 [0:3];
  integer pe_west_bind_p2 = 0;
  initial begin
    pe_west_bind_src2[0] = 8'h00;
    pe_west_bind_src2[1] = 8'h02;
    pe_west_bind_src2[2] = 8'h01;
    pe_west_bind_src2[3] = 8'h02;
  end
  assign pe_west_bind_dout[2] = pe_west_bind_src2[pe_west_bind_p2 < 4 ? pe_west_bind_p2 : 3];
  assign pe_west_bind_empty_n[2] = (pe_west_bind_p2 < 4);
  always @(posedge clk) if (rst_n && pe_west_bind_read[2] && pe_west_bind_empty_n[2]) pe_west_bind_p2 <= pe_west_bind_p2 + 1;
  reg [7:0] pe_west_bind_src3 [0:3];
  integer pe_west_bind_p3 = 0;
  initial begin
    pe_west_bind_src3[0] = 8'h01;
    pe_west_bind_src3[1] = 8'h01;
    pe_west_bind_src3[2] = 8'h02;
    pe_west_bind_src3[3] = 8'h02;
  end
  assign pe_west_bind_dout[3] = pe_west_bind_src3[pe_west_bind_p3 < 4 ? pe_west_bind_p3 : 3];
  assign pe_west_bind_empty_n[3] = (pe_west_bind_p3 < 4);
  always @(posedge clk) if (rst_n && pe_west_bind_read[3] && pe_west_bind_empty_n[3]) pe_west_bind_p3 <= pe_west_bind_p3 + 1;
  wire [7:0] pe_north_bind_dout [0:3];
  wire pe_north_bind_empty_n [0:3];
  wire pe_north_bind_read [0:3];
  reg [7:0] pe_north_bind_src0 [0:3];
  integer pe_north_bind_p0 = 0;
  initial begin
    pe_north_bind_src0[0] = 8'h01;
    pe_north_bind_src0[1] = 8'h00;
    pe_north_bind_src0[2] = 8'h01;
    pe_north_bind_src0[3] = 8'h02;
  end
  assign pe_north_bind_dout[0] = pe_north_bind_src0[pe_north_bind_p0 < 4 ? pe_north_bind_p0 : 3];
  assign pe_north_bind_empty_n[0] = (pe_north_bind_p0 < 4);
  always @(posedge clk) if (rst_n && pe_north_bind_read[0] && pe_north_bind_empty_n[0]) pe_north_bind_p0 <= pe_north_bind_p0 + 1;
  reg [7:0] pe_north_bind_src1 [0:3];
  integer pe_north_bind_p1 = 0;
  initial begin
    pe_north_bind_src1[0] = 8'h01;
    pe_north_bind_src1[1] = 8'h02;
    pe_north_bind_src1[2] = 8'h02;
    pe_north_bind_src1[3] = 8'h02;
  end
  assign pe_north_bind_dout[1] = pe_north_bind_src1[pe_north_bind_p1 < 4 ? pe_north_bind_p1 : 3];
  assign pe_north_bind_empty_n[1] = (pe_north_bind_p1 < 4);
  always @(posedge clk) if (rst_n && pe_north_bind_read[1] && pe_north_bind_empty_n[1]) pe_north_bind_p1 <= pe_north_bind_p1 + 1;
  reg [7:0] pe_north_bind_src2 [0:3];
  integer pe_north_bind_p2 = 0;
  initial begin
    pe_north_bind_src2[0] = 8'h01;
    pe_north_bind_src2[1] = 8'h02;
    pe_north_bind_src2[2] = 8'h01;
    pe_north_bind_src2[3] = 8'h02;
  end
  assign pe_north_bind_dout[2] = pe_north_bind_src2[pe_north_bind_p2 < 4 ? pe_north_bind_p2 : 3];
  assign pe_north_bind_empty_n[2] = (pe_north_bind_p2 < 4);
  always @(posedge clk) if (rst_n && pe_north_bind_read[2] && pe_north_bind_empty_n[2]) pe_north_bind_p2 <= pe_north_bind_p2 + 1;
  reg [7:0] pe_north_bind_src3 [0:3];
  integer pe_north_bind_p3 = 0;
  initial begin
    pe_north_bind_src3[0] = 8'h02;
    pe_north_bind_src3[1] = 8'h00;
    pe_north_bind_src3[2] = 8'h00;
    pe_north_bind_src3[3] = 8'h00;
  end
  assign pe_north_bind_dout[3] = pe_north_bind_src3[pe_north_bind_p3 < 4 ? pe_north_bind_p3 : 3];
  assign pe_north_bind_empty_n[3] = (pe_north_bind_p3 < 4);
  always @(posedge clk) if (rst_n && pe_north_bind_read[3] && pe_north_bind_empty_n[3]) pe_north_bind_p3 <= pe_north_bind_p3 + 1;
  wire [31:0] pe_c_mem_din [0:15];
  wire pe_c_mem_write [0:15];
  wire pe_c_mem_full_n [0:15];
  reg [31:0] pe_c_mem_exp0 [0:0];
  integer pe_c_mem_q0 = 0;
  initial begin
    pe_c_mem_exp0[0] = 32'h00000003;
  end
  assign pe_c_mem_full_n[0] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[0]) begin
    if (pe_c_mem_q0 < 1) begin
      if (pe_c_mem_din[0] !== pe_c_mem_exp0[pe_c_mem_q0]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 0, pe_c_mem_q0, pe_c_mem_din[0], pe_c_mem_exp0[pe_c_mem_q0]);
      end
      pe_c_mem_q0 <= pe_c_mem_q0 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 0);
    end
  end
  reg [31:0] pe_c_mem_exp1 [0:0];
  integer pe_c_mem_q1 = 0;
  initial begin
    pe_c_mem_exp1[0] = 32'h00000006;
  end
  assign pe_c_mem_full_n[1] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[1]) begin
    if (pe_c_mem_q1 < 1) begin
      if (pe_c_mem_din[1] !== pe_c_mem_exp1[pe_c_mem_q1]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 1, pe_c_mem_q1, pe_c_mem_din[1], pe_c_mem_exp1[pe_c_mem_q1]);
      end
      pe_c_mem_q1 <= pe_c_mem_q1 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 1);
    end
  end
  reg [31:0] pe_c_mem_exp2 [0:0];
  integer pe_c_mem_q2 = 0;
  initial begin
    pe_c_mem_exp2[0] = 32'h00000005;
  end
  assign pe_c_mem_full_n[2] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[2]) begin
    if (pe_c_mem_q2 < 1) begin
      if (pe_c_mem_din[2] !== pe_c_mem_exp2[pe_c_mem_q2]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 2, pe_c_mem_q2, pe_c_mem_din[2], pe_c_mem_exp2[pe_c_mem_q2]);
      end
      pe_c_mem_q2 <= pe_c_mem_q2 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 2);
    end
  end
  reg [31:0] pe_c_mem_exp3 [0:0];
  integer pe_c_mem_q3 = 0;
  initial begin
    pe_c_mem_exp3[0] = 32'h00000004;
  end
  assign pe_c_mem_full_n[3] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[3]) begin
    if (pe_c_mem_q3 < 1) begin
      if (pe_c_mem_din[3] !== pe_c_mem_exp3[pe_c_mem_q3]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 3, pe_c_mem_q3, pe_c_mem_din[3], pe_c_mem_exp3[pe_c_mem_q3]);
      end
      pe_c_mem_q3 <= pe_c_mem_q3 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 3);
    end
  end
  reg [31:0] pe_c_mem_exp4 [0:0];
  integer pe_c_mem_q4 = 0;
  initial begin
    pe_c_mem_exp4[0] = 32'h00000000;
  end
  assign pe_c_mem_full_n[4] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[4]) begin
    if (pe_c_mem_q4 < 1) begin
      if (pe_c_mem_din[4] !== pe_c_mem_exp4[pe_c_mem_q4]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 4, pe_c_mem_q4, pe_c_mem_din[4], pe_c_mem_exp4[pe_c_mem_q4]);
      end
      pe_c_mem_q4 <= pe_c_mem_q4 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 4);
    end
  end
  reg [31:0] pe_c_mem_exp5 [0:0];
  integer pe_c_mem_q5 = 0;
  initial begin
    pe_c_mem_exp5[0] = 32'h00000000;
  end
  assign pe_c_mem_full_n[5] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[5]) begin
    if (pe_c_mem_q5 < 1) begin
      if (pe_c_mem_din[5] !== pe_c_mem_exp5[pe_c_mem_q5]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 5, pe_c_mem_q5, pe_c_mem_din[5], pe_c_mem_exp5[pe_c_mem_q5]);
      end
      pe_c_mem_q5 <= pe_c_mem_q5 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 5);
    end
  end
  reg [31:0] pe_c_mem_exp6 [0:0];
  integer pe_c_mem_q6 = 0;
  initial begin
    pe_c_mem_exp6[0] = 32'h00000000;
  end
  assign pe_c_mem_full_n[6] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[6]) begin
    if (pe_c_mem_q6 < 1) begin
      if (pe_c_mem_din[6] !== pe_c_mem_exp6[pe_c_mem_q6]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 6, pe_c_mem_q6, pe_c_mem_din[6], pe_c_mem_exp6[pe_c_mem_q6]);
      end
      pe_c_mem_q6 <= pe_c_mem_q6 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 6);
    end
  end
  reg [31:0] pe_c_mem_exp7 [0:0];
  integer pe_c_mem_q7 = 0;
  initial begin
    pe_c_mem_exp7[0] = 32'h00000000;
  end
  assign pe_c_mem_full_n[7] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[7]) begin
    if (pe_c_mem_q7 < 1) begin
      if (pe_c_mem_din[7] !== pe_c_mem_exp7[pe_c_mem_q7]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 7, pe_c_mem_q7, pe_c_mem_din[7], pe_c_mem_exp7[pe_c_mem_q7]);
      end
      pe_c_mem_q7 <= pe_c_mem_q7 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 7);
    end
  end
  reg [31:0] pe_c_mem_exp8 [0:0];
  integer pe_c_mem_q8 = 0;
  initial begin
    pe_c_mem_exp8[0] = 32'h00000005;
  end
  assign pe_c_mem_full_n[8] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[8]) begin
    if (pe_c_mem_q8 < 1) begin
      if (pe_c_mem_din[8] !== pe_c_mem_exp8[pe_c_mem_q8]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 8, pe_c_mem_q8, pe_c_mem_din[8], pe_c_mem_exp8[pe_c_mem_q8]);
      end
      pe_c_mem_q8 <= pe_c_mem_q8 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 8);
    end
  end
  reg [31:0] pe_c_mem_exp9 [0:0];
  integer pe_c_mem_q9 = 0;
  initial begin
    pe_c_mem_exp9[0] = 32'h0000000a;
  end
  assign pe_c_mem_full_n[9] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[9]) begin
    if (pe_c_mem_q9 < 1) begin
      if (pe_c_mem_din[9] !== pe_c_mem_exp9[pe_c_mem_q9]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 9, pe_c_mem_q9, pe_c_mem_din[9], pe_c_mem_exp9[pe_c_mem_q9]);
      end
      pe_c_mem_q9 <= pe_c_mem_q9 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 9);
    end
  end
  reg [31:0] pe_c_mem_exp10 [0:0];
  integer pe_c_mem_q10 = 0;
  initial begin
    pe_c_mem_exp10[0] = 32'h00000009;
  end
  assign pe_c_mem_full_n[10] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[10]) begin
    if (pe_c_mem_q10 < 1) begin
      if (pe_c_mem_din[10] !== pe_c_mem_exp10[pe_c_mem_q10]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 10, pe_c_mem_q10, pe_c_mem_din[10], pe_c_mem_exp10[pe_c_mem_q10]);
      end
      pe_c_mem_q10 <= pe_c_mem_q10 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 10);
    end
  end
  reg [31:0] pe_c_mem_exp11 [0:0];
  integer pe_c_mem_q11 = 0;
  initial begin
    pe_c_mem_exp11[0] = 32'h00000000;
  end
  assign pe_c_mem_full_n[11] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[11]) begin
    if (pe_c_mem_q11 < 1) begin
      if (pe_c_mem_din[11] !== pe_c_mem_exp11[pe_c_mem_q11]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 11, pe_c_mem_q11, pe_c_mem_din[11], pe_c_mem_exp11[pe_c_mem_q11]);
      end
      pe_c_mem_q11 <= pe_c_mem_q11 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 11);
    end
  end
  reg [31:0] pe_c_mem_exp12 [0:0];
  integer pe_c_mem_q12 = 0;
  initial begin
    pe_c_mem_exp12[0] = 32'h00000007;
  end
  assign pe_c_mem_full_n[12] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[12]) begin
    if (pe_c_mem_q12 < 1) begin
      if (pe_c_mem_din[12] !== pe_c_mem_exp12[pe_c_mem_q12]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 12, pe_c_mem_q12, pe_c_mem_din[12], pe_c_mem_exp12[pe_c_mem_q12]);
      end
      pe_c_mem_q12 <= pe_c_mem_q12 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 12);
    end
  end
  reg [31:0] pe_c_mem_exp13 [0:0];
  integer pe_c_mem_q13 = 0;
  initial begin
    pe_c_mem_exp13[0] = 32'h0000000b;
  end
  assign pe_c_mem_full_n[13] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[13]) begin
    if (pe_c_mem_q13 < 1) begin
      if (pe_c_mem_din[13] !== pe_c_mem_exp13[pe_c_mem_q13]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 13, pe_c_mem_q13, pe_c_mem_din[13], pe_c_mem_exp13[pe_c_mem_q13]);
      end
      pe_c_mem_q13 <= pe_c_mem_q13 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 13);
    end
  end
  reg [31:0] pe_c_mem_exp14 [0:0];
  integer pe_c_mem_q14 = 0;
  initial begin
    pe_c_mem_exp14[0] = 32'h00000009;
  end
  assign pe_c_mem_full_n[14] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[14]) begin
    if (pe_c_mem_q14 < 1) begin
      if (pe_c_mem_din[14] !== pe_c_mem_exp14[pe_c_mem_q14]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 14, pe_c_mem_q14, pe_c_mem_din[14], pe_c_mem_exp14[pe_c_mem_q14]);
      end
      pe_c_mem_q14 <= pe_c_mem_q14 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 14);
    end
  end
  reg [31:0] pe_c_mem_exp15 [0:0];
  integer pe_c_mem_q15 = 0;
  initial begin
    pe_c_mem_exp15[0] = 32'h00000002;
  end
  assign pe_c_mem_full_n[15] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[15]) begin
    if (pe_c_mem_q15 < 1) begin
      if (pe_c_mem_din[15] !== pe_c_mem_exp15[pe_c_mem_q15]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_mem[%0d] step %0d: got %h want %h",
                 15, pe_c_mem_q15, pe_c_mem_din[15], pe_c_mem_exp15[pe_c_mem_q15]);
      end
      pe_c_mem_q15 <= pe_c_mem_q15 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_mem[%0d]", 15);
    end
  end
  spmw_top dut (.ap_clk(clk), .ap_rst_n(rst_n), .pe_west_bind_dout(pe_west_bind_dout), .pe_west_bind_empty_n(pe_west_bind_empty_n), .pe_west_bind_read(pe_west_bind_read), .pe_north_bind_dout(pe_north_bind_dout), .pe_north_bind_empty_n(pe_north_bind_empty_n), .pe_north_bind_read(pe_north_bind_read), .pe_c_mem_din(pe_c_mem_din), .pe_c_mem_write(pe_c_mem_write), .pe_c_mem_full_n(pe_c_mem_full_n));
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
