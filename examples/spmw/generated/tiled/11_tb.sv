`timescale 1ns/1ps

module tb;
  reg clk = 0, rst_n = 0;
  always #5 clk = ~clk;
  integer errors = 0;
  integer produced = 0;
  localparam integer TOTAL = 16;
  wire [31:0] pe_west_bind_dout [0:1];
  wire pe_west_bind_empty_n [0:1];
  wire pe_west_bind_read [0:1];
  reg [31:0] pe_west_bind_src0 [0:3];
  integer pe_west_bind_p0 = 0;
  initial begin
    pe_west_bind_src0[0] = 32'h40000000;
    pe_west_bind_src0[1] = 32'h3f800000;
    pe_west_bind_src0[2] = 32'h3f800000;
    pe_west_bind_src0[3] = 32'h00000000;
  end
  assign pe_west_bind_dout[0] = pe_west_bind_src0[pe_west_bind_p0 < 4 ? pe_west_bind_p0 : 3];
  assign pe_west_bind_empty_n[0] = (pe_west_bind_p0 < 4);
  always @(posedge clk) if (rst_n && pe_west_bind_read[0] && pe_west_bind_empty_n[0]) pe_west_bind_p0 <= pe_west_bind_p0 + 1;
  reg [31:0] pe_west_bind_src1 [0:3];
  integer pe_west_bind_p1 = 0;
  initial begin
    pe_west_bind_src1[0] = 32'h00000000;
    pe_west_bind_src1[1] = 32'h00000000;
    pe_west_bind_src1[2] = 32'h00000000;
    pe_west_bind_src1[3] = 32'h00000000;
  end
  assign pe_west_bind_dout[1] = pe_west_bind_src1[pe_west_bind_p1 < 4 ? pe_west_bind_p1 : 3];
  assign pe_west_bind_empty_n[1] = (pe_west_bind_p1 < 4);
  always @(posedge clk) if (rst_n && pe_west_bind_read[1] && pe_west_bind_empty_n[1]) pe_west_bind_p1 <= pe_west_bind_p1 + 1;
  wire [31:0] pe_north_bind_dout [0:1];
  wire pe_north_bind_empty_n [0:1];
  wire pe_north_bind_read [0:1];
  reg [31:0] pe_north_bind_src0 [0:3];
  integer pe_north_bind_p0 = 0;
  initial begin
    pe_north_bind_src0[0] = 32'h3f800000;
    pe_north_bind_src0[1] = 32'h00000000;
    pe_north_bind_src0[2] = 32'h3f800000;
    pe_north_bind_src0[3] = 32'h40000000;
  end
  assign pe_north_bind_dout[0] = pe_north_bind_src0[pe_north_bind_p0 < 4 ? pe_north_bind_p0 : 3];
  assign pe_north_bind_empty_n[0] = (pe_north_bind_p0 < 4);
  always @(posedge clk) if (rst_n && pe_north_bind_read[0] && pe_north_bind_empty_n[0]) pe_north_bind_p0 <= pe_north_bind_p0 + 1;
  reg [31:0] pe_north_bind_src1 [0:3];
  integer pe_north_bind_p1 = 0;
  initial begin
    pe_north_bind_src1[0] = 32'h3f800000;
    pe_north_bind_src1[1] = 32'h40000000;
    pe_north_bind_src1[2] = 32'h40000000;
    pe_north_bind_src1[3] = 32'h40000000;
  end
  assign pe_north_bind_dout[1] = pe_north_bind_src1[pe_north_bind_p1 < 4 ? pe_north_bind_p1 : 3];
  assign pe_north_bind_empty_n[1] = (pe_north_bind_p1 < 4);
  always @(posedge clk) if (rst_n && pe_north_bind_read[1] && pe_north_bind_empty_n[1]) pe_north_bind_p1 <= pe_north_bind_p1 + 1;
  wire [31:0] pe_1_west_bind_dout [0:1];
  wire pe_1_west_bind_empty_n [0:1];
  wire pe_1_west_bind_read [0:1];
  reg [31:0] pe_1_west_bind_src0 [0:3];
  integer pe_1_west_bind_p0 = 0;
  initial begin
    pe_1_west_bind_src0[0] = 32'h40000000;
    pe_1_west_bind_src0[1] = 32'h3f800000;
    pe_1_west_bind_src0[2] = 32'h3f800000;
    pe_1_west_bind_src0[3] = 32'h00000000;
  end
  assign pe_1_west_bind_dout[0] = pe_1_west_bind_src0[pe_1_west_bind_p0 < 4 ? pe_1_west_bind_p0 : 3];
  assign pe_1_west_bind_empty_n[0] = (pe_1_west_bind_p0 < 4);
  always @(posedge clk) if (rst_n && pe_1_west_bind_read[0] && pe_1_west_bind_empty_n[0]) pe_1_west_bind_p0 <= pe_1_west_bind_p0 + 1;
  reg [31:0] pe_1_west_bind_src1 [0:3];
  integer pe_1_west_bind_p1 = 0;
  initial begin
    pe_1_west_bind_src1[0] = 32'h00000000;
    pe_1_west_bind_src1[1] = 32'h00000000;
    pe_1_west_bind_src1[2] = 32'h00000000;
    pe_1_west_bind_src1[3] = 32'h00000000;
  end
  assign pe_1_west_bind_dout[1] = pe_1_west_bind_src1[pe_1_west_bind_p1 < 4 ? pe_1_west_bind_p1 : 3];
  assign pe_1_west_bind_empty_n[1] = (pe_1_west_bind_p1 < 4);
  always @(posedge clk) if (rst_n && pe_1_west_bind_read[1] && pe_1_west_bind_empty_n[1]) pe_1_west_bind_p1 <= pe_1_west_bind_p1 + 1;
  wire [31:0] pe_1_north_bind_dout [0:1];
  wire pe_1_north_bind_empty_n [0:1];
  wire pe_1_north_bind_read [0:1];
  reg [31:0] pe_1_north_bind_src0 [0:3];
  integer pe_1_north_bind_p0 = 0;
  initial begin
    pe_1_north_bind_src0[0] = 32'h3f800000;
    pe_1_north_bind_src0[1] = 32'h40000000;
    pe_1_north_bind_src0[2] = 32'h3f800000;
    pe_1_north_bind_src0[3] = 32'h40000000;
  end
  assign pe_1_north_bind_dout[0] = pe_1_north_bind_src0[pe_1_north_bind_p0 < 4 ? pe_1_north_bind_p0 : 3];
  assign pe_1_north_bind_empty_n[0] = (pe_1_north_bind_p0 < 4);
  always @(posedge clk) if (rst_n && pe_1_north_bind_read[0] && pe_1_north_bind_empty_n[0]) pe_1_north_bind_p0 <= pe_1_north_bind_p0 + 1;
  reg [31:0] pe_1_north_bind_src1 [0:3];
  integer pe_1_north_bind_p1 = 0;
  initial begin
    pe_1_north_bind_src1[0] = 32'h40000000;
    pe_1_north_bind_src1[1] = 32'h00000000;
    pe_1_north_bind_src1[2] = 32'h00000000;
    pe_1_north_bind_src1[3] = 32'h00000000;
  end
  assign pe_1_north_bind_dout[1] = pe_1_north_bind_src1[pe_1_north_bind_p1 < 4 ? pe_1_north_bind_p1 : 3];
  assign pe_1_north_bind_empty_n[1] = (pe_1_north_bind_p1 < 4);
  always @(posedge clk) if (rst_n && pe_1_north_bind_read[1] && pe_1_north_bind_empty_n[1]) pe_1_north_bind_p1 <= pe_1_north_bind_p1 + 1;
  wire [31:0] pe_2_west_bind_dout [0:1];
  wire pe_2_west_bind_empty_n [0:1];
  wire pe_2_west_bind_read [0:1];
  reg [31:0] pe_2_west_bind_src0 [0:3];
  integer pe_2_west_bind_p0 = 0;
  initial begin
    pe_2_west_bind_src0[0] = 32'h00000000;
    pe_2_west_bind_src0[1] = 32'h40000000;
    pe_2_west_bind_src0[2] = 32'h3f800000;
    pe_2_west_bind_src0[3] = 32'h40000000;
  end
  assign pe_2_west_bind_dout[0] = pe_2_west_bind_src0[pe_2_west_bind_p0 < 4 ? pe_2_west_bind_p0 : 3];
  assign pe_2_west_bind_empty_n[0] = (pe_2_west_bind_p0 < 4);
  always @(posedge clk) if (rst_n && pe_2_west_bind_read[0] && pe_2_west_bind_empty_n[0]) pe_2_west_bind_p0 <= pe_2_west_bind_p0 + 1;
  reg [31:0] pe_2_west_bind_src1 [0:3];
  integer pe_2_west_bind_p1 = 0;
  initial begin
    pe_2_west_bind_src1[0] = 32'h3f800000;
    pe_2_west_bind_src1[1] = 32'h3f800000;
    pe_2_west_bind_src1[2] = 32'h40000000;
    pe_2_west_bind_src1[3] = 32'h40000000;
  end
  assign pe_2_west_bind_dout[1] = pe_2_west_bind_src1[pe_2_west_bind_p1 < 4 ? pe_2_west_bind_p1 : 3];
  assign pe_2_west_bind_empty_n[1] = (pe_2_west_bind_p1 < 4);
  always @(posedge clk) if (rst_n && pe_2_west_bind_read[1] && pe_2_west_bind_empty_n[1]) pe_2_west_bind_p1 <= pe_2_west_bind_p1 + 1;
  wire [31:0] pe_2_north_bind_dout [0:1];
  wire pe_2_north_bind_empty_n [0:1];
  wire pe_2_north_bind_read [0:1];
  reg [31:0] pe_2_north_bind_src0 [0:3];
  integer pe_2_north_bind_p0 = 0;
  initial begin
    pe_2_north_bind_src0[0] = 32'h3f800000;
    pe_2_north_bind_src0[1] = 32'h00000000;
    pe_2_north_bind_src0[2] = 32'h3f800000;
    pe_2_north_bind_src0[3] = 32'h40000000;
  end
  assign pe_2_north_bind_dout[0] = pe_2_north_bind_src0[pe_2_north_bind_p0 < 4 ? pe_2_north_bind_p0 : 3];
  assign pe_2_north_bind_empty_n[0] = (pe_2_north_bind_p0 < 4);
  always @(posedge clk) if (rst_n && pe_2_north_bind_read[0] && pe_2_north_bind_empty_n[0]) pe_2_north_bind_p0 <= pe_2_north_bind_p0 + 1;
  reg [31:0] pe_2_north_bind_src1 [0:3];
  integer pe_2_north_bind_p1 = 0;
  initial begin
    pe_2_north_bind_src1[0] = 32'h3f800000;
    pe_2_north_bind_src1[1] = 32'h40000000;
    pe_2_north_bind_src1[2] = 32'h40000000;
    pe_2_north_bind_src1[3] = 32'h40000000;
  end
  assign pe_2_north_bind_dout[1] = pe_2_north_bind_src1[pe_2_north_bind_p1 < 4 ? pe_2_north_bind_p1 : 3];
  assign pe_2_north_bind_empty_n[1] = (pe_2_north_bind_p1 < 4);
  always @(posedge clk) if (rst_n && pe_2_north_bind_read[1] && pe_2_north_bind_empty_n[1]) pe_2_north_bind_p1 <= pe_2_north_bind_p1 + 1;
  wire [31:0] pe_3_west_bind_dout [0:1];
  wire pe_3_west_bind_empty_n [0:1];
  wire pe_3_west_bind_read [0:1];
  reg [31:0] pe_3_west_bind_src0 [0:3];
  integer pe_3_west_bind_p0 = 0;
  initial begin
    pe_3_west_bind_src0[0] = 32'h00000000;
    pe_3_west_bind_src0[1] = 32'h40000000;
    pe_3_west_bind_src0[2] = 32'h3f800000;
    pe_3_west_bind_src0[3] = 32'h40000000;
  end
  assign pe_3_west_bind_dout[0] = pe_3_west_bind_src0[pe_3_west_bind_p0 < 4 ? pe_3_west_bind_p0 : 3];
  assign pe_3_west_bind_empty_n[0] = (pe_3_west_bind_p0 < 4);
  always @(posedge clk) if (rst_n && pe_3_west_bind_read[0] && pe_3_west_bind_empty_n[0]) pe_3_west_bind_p0 <= pe_3_west_bind_p0 + 1;
  reg [31:0] pe_3_west_bind_src1 [0:3];
  integer pe_3_west_bind_p1 = 0;
  initial begin
    pe_3_west_bind_src1[0] = 32'h3f800000;
    pe_3_west_bind_src1[1] = 32'h3f800000;
    pe_3_west_bind_src1[2] = 32'h40000000;
    pe_3_west_bind_src1[3] = 32'h40000000;
  end
  assign pe_3_west_bind_dout[1] = pe_3_west_bind_src1[pe_3_west_bind_p1 < 4 ? pe_3_west_bind_p1 : 3];
  assign pe_3_west_bind_empty_n[1] = (pe_3_west_bind_p1 < 4);
  always @(posedge clk) if (rst_n && pe_3_west_bind_read[1] && pe_3_west_bind_empty_n[1]) pe_3_west_bind_p1 <= pe_3_west_bind_p1 + 1;
  wire [31:0] pe_3_north_bind_dout [0:1];
  wire pe_3_north_bind_empty_n [0:1];
  wire pe_3_north_bind_read [0:1];
  reg [31:0] pe_3_north_bind_src0 [0:3];
  integer pe_3_north_bind_p0 = 0;
  initial begin
    pe_3_north_bind_src0[0] = 32'h3f800000;
    pe_3_north_bind_src0[1] = 32'h40000000;
    pe_3_north_bind_src0[2] = 32'h3f800000;
    pe_3_north_bind_src0[3] = 32'h40000000;
  end
  assign pe_3_north_bind_dout[0] = pe_3_north_bind_src0[pe_3_north_bind_p0 < 4 ? pe_3_north_bind_p0 : 3];
  assign pe_3_north_bind_empty_n[0] = (pe_3_north_bind_p0 < 4);
  always @(posedge clk) if (rst_n && pe_3_north_bind_read[0] && pe_3_north_bind_empty_n[0]) pe_3_north_bind_p0 <= pe_3_north_bind_p0 + 1;
  reg [31:0] pe_3_north_bind_src1 [0:3];
  integer pe_3_north_bind_p1 = 0;
  initial begin
    pe_3_north_bind_src1[0] = 32'h40000000;
    pe_3_north_bind_src1[1] = 32'h00000000;
    pe_3_north_bind_src1[2] = 32'h00000000;
    pe_3_north_bind_src1[3] = 32'h00000000;
  end
  assign pe_3_north_bind_dout[1] = pe_3_north_bind_src1[pe_3_north_bind_p1 < 4 ? pe_3_north_bind_p1 : 3];
  assign pe_3_north_bind_empty_n[1] = (pe_3_north_bind_p1 < 4);
  always @(posedge clk) if (rst_n && pe_3_north_bind_read[1] && pe_3_north_bind_empty_n[1]) pe_3_north_bind_p1 <= pe_3_north_bind_p1 + 1;
  wire [31:0] pe_c_mem_din [0:3];
  wire pe_c_mem_write [0:3];
  wire pe_c_mem_full_n [0:3];
  reg [31:0] pe_c_mem_exp0 [0:0];
  integer pe_c_mem_q0 = 0;
  initial begin
    pe_c_mem_exp0[0] = 32'h40400000;
  end
  assign pe_c_mem_full_n[0] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[0]) begin
    if (pe_c_mem_q0 < 1) begin
      if (((($bitstoshortreal(pe_c_mem_din[0][31:0])) - ($bitstoshortreal(pe_c_mem_exp0[pe_c_mem_q0][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_c_mem_exp0[pe_c_mem_q0][31:0])) < 0.0) ? -($bitstoshortreal(pe_c_mem_exp0[pe_c_mem_q0][31:0])) : ($bitstoshortreal(pe_c_mem_exp0[pe_c_mem_q0][31:0])))) || ((($bitstoshortreal(pe_c_mem_exp0[pe_c_mem_q0][31:0])) - ($bitstoshortreal(pe_c_mem_din[0][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_c_mem_exp0[pe_c_mem_q0][31:0])) < 0.0) ? -($bitstoshortreal(pe_c_mem_exp0[pe_c_mem_q0][31:0])) : ($bitstoshortreal(pe_c_mem_exp0[pe_c_mem_q0][31:0])))))) begin
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
    pe_c_mem_exp1[0] = 32'h40c00000;
  end
  assign pe_c_mem_full_n[1] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[1]) begin
    if (pe_c_mem_q1 < 1) begin
      if (((($bitstoshortreal(pe_c_mem_din[1][31:0])) - ($bitstoshortreal(pe_c_mem_exp1[pe_c_mem_q1][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_c_mem_exp1[pe_c_mem_q1][31:0])) < 0.0) ? -($bitstoshortreal(pe_c_mem_exp1[pe_c_mem_q1][31:0])) : ($bitstoshortreal(pe_c_mem_exp1[pe_c_mem_q1][31:0])))) || ((($bitstoshortreal(pe_c_mem_exp1[pe_c_mem_q1][31:0])) - ($bitstoshortreal(pe_c_mem_din[1][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_c_mem_exp1[pe_c_mem_q1][31:0])) < 0.0) ? -($bitstoshortreal(pe_c_mem_exp1[pe_c_mem_q1][31:0])) : ($bitstoshortreal(pe_c_mem_exp1[pe_c_mem_q1][31:0])))))) begin
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
    pe_c_mem_exp2[0] = 32'h00000000;
  end
  assign pe_c_mem_full_n[2] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[2]) begin
    if (pe_c_mem_q2 < 1) begin
      if (((($bitstoshortreal(pe_c_mem_din[2][31:0])) - ($bitstoshortreal(pe_c_mem_exp2[pe_c_mem_q2][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_c_mem_exp2[pe_c_mem_q2][31:0])) < 0.0) ? -($bitstoshortreal(pe_c_mem_exp2[pe_c_mem_q2][31:0])) : ($bitstoshortreal(pe_c_mem_exp2[pe_c_mem_q2][31:0])))) || ((($bitstoshortreal(pe_c_mem_exp2[pe_c_mem_q2][31:0])) - ($bitstoshortreal(pe_c_mem_din[2][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_c_mem_exp2[pe_c_mem_q2][31:0])) < 0.0) ? -($bitstoshortreal(pe_c_mem_exp2[pe_c_mem_q2][31:0])) : ($bitstoshortreal(pe_c_mem_exp2[pe_c_mem_q2][31:0])))))) begin
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
    pe_c_mem_exp3[0] = 32'h00000000;
  end
  assign pe_c_mem_full_n[3] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_mem_write[3]) begin
    if (pe_c_mem_q3 < 1) begin
      if (((($bitstoshortreal(pe_c_mem_din[3][31:0])) - ($bitstoshortreal(pe_c_mem_exp3[pe_c_mem_q3][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_c_mem_exp3[pe_c_mem_q3][31:0])) < 0.0) ? -($bitstoshortreal(pe_c_mem_exp3[pe_c_mem_q3][31:0])) : ($bitstoshortreal(pe_c_mem_exp3[pe_c_mem_q3][31:0])))) || ((($bitstoshortreal(pe_c_mem_exp3[pe_c_mem_q3][31:0])) - ($bitstoshortreal(pe_c_mem_din[3][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_c_mem_exp3[pe_c_mem_q3][31:0])) < 0.0) ? -($bitstoshortreal(pe_c_mem_exp3[pe_c_mem_q3][31:0])) : ($bitstoshortreal(pe_c_mem_exp3[pe_c_mem_q3][31:0])))))) begin
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
  wire [31:0] pe_1_c_mem_din [0:3];
  wire pe_1_c_mem_write [0:3];
  wire pe_1_c_mem_full_n [0:3];
  reg [31:0] pe_1_c_mem_exp0 [0:0];
  integer pe_1_c_mem_q0 = 0;
  initial begin
    pe_1_c_mem_exp0[0] = 32'h40a00000;
  end
  assign pe_1_c_mem_full_n[0] = 1'b1;
  always @(posedge clk) if (rst_n && pe_1_c_mem_write[0]) begin
    if (pe_1_c_mem_q0 < 1) begin
      if (((($bitstoshortreal(pe_1_c_mem_din[0][31:0])) - ($bitstoshortreal(pe_1_c_mem_exp0[pe_1_c_mem_q0][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_1_c_mem_exp0[pe_1_c_mem_q0][31:0])) < 0.0) ? -($bitstoshortreal(pe_1_c_mem_exp0[pe_1_c_mem_q0][31:0])) : ($bitstoshortreal(pe_1_c_mem_exp0[pe_1_c_mem_q0][31:0])))) || ((($bitstoshortreal(pe_1_c_mem_exp0[pe_1_c_mem_q0][31:0])) - ($bitstoshortreal(pe_1_c_mem_din[0][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_1_c_mem_exp0[pe_1_c_mem_q0][31:0])) < 0.0) ? -($bitstoshortreal(pe_1_c_mem_exp0[pe_1_c_mem_q0][31:0])) : ($bitstoshortreal(pe_1_c_mem_exp0[pe_1_c_mem_q0][31:0])))))) begin
        errors = errors + 1;
        $display("MISMATCH pe_1_c_mem[%0d] step %0d: got %h want %h",
                 0, pe_1_c_mem_q0, pe_1_c_mem_din[0], pe_1_c_mem_exp0[pe_1_c_mem_q0]);
      end
      pe_1_c_mem_q0 <= pe_1_c_mem_q0 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_1_c_mem[%0d]", 0);
    end
  end
  reg [31:0] pe_1_c_mem_exp1 [0:0];
  integer pe_1_c_mem_q1 = 0;
  initial begin
    pe_1_c_mem_exp1[0] = 32'h40800000;
  end
  assign pe_1_c_mem_full_n[1] = 1'b1;
  always @(posedge clk) if (rst_n && pe_1_c_mem_write[1]) begin
    if (pe_1_c_mem_q1 < 1) begin
      if (((($bitstoshortreal(pe_1_c_mem_din[1][31:0])) - ($bitstoshortreal(pe_1_c_mem_exp1[pe_1_c_mem_q1][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_1_c_mem_exp1[pe_1_c_mem_q1][31:0])) < 0.0) ? -($bitstoshortreal(pe_1_c_mem_exp1[pe_1_c_mem_q1][31:0])) : ($bitstoshortreal(pe_1_c_mem_exp1[pe_1_c_mem_q1][31:0])))) || ((($bitstoshortreal(pe_1_c_mem_exp1[pe_1_c_mem_q1][31:0])) - ($bitstoshortreal(pe_1_c_mem_din[1][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_1_c_mem_exp1[pe_1_c_mem_q1][31:0])) < 0.0) ? -($bitstoshortreal(pe_1_c_mem_exp1[pe_1_c_mem_q1][31:0])) : ($bitstoshortreal(pe_1_c_mem_exp1[pe_1_c_mem_q1][31:0])))))) begin
        errors = errors + 1;
        $display("MISMATCH pe_1_c_mem[%0d] step %0d: got %h want %h",
                 1, pe_1_c_mem_q1, pe_1_c_mem_din[1], pe_1_c_mem_exp1[pe_1_c_mem_q1]);
      end
      pe_1_c_mem_q1 <= pe_1_c_mem_q1 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_1_c_mem[%0d]", 1);
    end
  end
  reg [31:0] pe_1_c_mem_exp2 [0:0];
  integer pe_1_c_mem_q2 = 0;
  initial begin
    pe_1_c_mem_exp2[0] = 32'h00000000;
  end
  assign pe_1_c_mem_full_n[2] = 1'b1;
  always @(posedge clk) if (rst_n && pe_1_c_mem_write[2]) begin
    if (pe_1_c_mem_q2 < 1) begin
      if (((($bitstoshortreal(pe_1_c_mem_din[2][31:0])) - ($bitstoshortreal(pe_1_c_mem_exp2[pe_1_c_mem_q2][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_1_c_mem_exp2[pe_1_c_mem_q2][31:0])) < 0.0) ? -($bitstoshortreal(pe_1_c_mem_exp2[pe_1_c_mem_q2][31:0])) : ($bitstoshortreal(pe_1_c_mem_exp2[pe_1_c_mem_q2][31:0])))) || ((($bitstoshortreal(pe_1_c_mem_exp2[pe_1_c_mem_q2][31:0])) - ($bitstoshortreal(pe_1_c_mem_din[2][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_1_c_mem_exp2[pe_1_c_mem_q2][31:0])) < 0.0) ? -($bitstoshortreal(pe_1_c_mem_exp2[pe_1_c_mem_q2][31:0])) : ($bitstoshortreal(pe_1_c_mem_exp2[pe_1_c_mem_q2][31:0])))))) begin
        errors = errors + 1;
        $display("MISMATCH pe_1_c_mem[%0d] step %0d: got %h want %h",
                 2, pe_1_c_mem_q2, pe_1_c_mem_din[2], pe_1_c_mem_exp2[pe_1_c_mem_q2]);
      end
      pe_1_c_mem_q2 <= pe_1_c_mem_q2 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_1_c_mem[%0d]", 2);
    end
  end
  reg [31:0] pe_1_c_mem_exp3 [0:0];
  integer pe_1_c_mem_q3 = 0;
  initial begin
    pe_1_c_mem_exp3[0] = 32'h00000000;
  end
  assign pe_1_c_mem_full_n[3] = 1'b1;
  always @(posedge clk) if (rst_n && pe_1_c_mem_write[3]) begin
    if (pe_1_c_mem_q3 < 1) begin
      if (((($bitstoshortreal(pe_1_c_mem_din[3][31:0])) - ($bitstoshortreal(pe_1_c_mem_exp3[pe_1_c_mem_q3][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_1_c_mem_exp3[pe_1_c_mem_q3][31:0])) < 0.0) ? -($bitstoshortreal(pe_1_c_mem_exp3[pe_1_c_mem_q3][31:0])) : ($bitstoshortreal(pe_1_c_mem_exp3[pe_1_c_mem_q3][31:0])))) || ((($bitstoshortreal(pe_1_c_mem_exp3[pe_1_c_mem_q3][31:0])) - ($bitstoshortreal(pe_1_c_mem_din[3][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_1_c_mem_exp3[pe_1_c_mem_q3][31:0])) < 0.0) ? -($bitstoshortreal(pe_1_c_mem_exp3[pe_1_c_mem_q3][31:0])) : ($bitstoshortreal(pe_1_c_mem_exp3[pe_1_c_mem_q3][31:0])))))) begin
        errors = errors + 1;
        $display("MISMATCH pe_1_c_mem[%0d] step %0d: got %h want %h",
                 3, pe_1_c_mem_q3, pe_1_c_mem_din[3], pe_1_c_mem_exp3[pe_1_c_mem_q3]);
      end
      pe_1_c_mem_q3 <= pe_1_c_mem_q3 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_1_c_mem[%0d]", 3);
    end
  end
  wire [31:0] pe_2_c_mem_din [0:3];
  wire pe_2_c_mem_write [0:3];
  wire pe_2_c_mem_full_n [0:3];
  reg [31:0] pe_2_c_mem_exp0 [0:0];
  integer pe_2_c_mem_q0 = 0;
  initial begin
    pe_2_c_mem_exp0[0] = 32'h40a00000;
  end
  assign pe_2_c_mem_full_n[0] = 1'b1;
  always @(posedge clk) if (rst_n && pe_2_c_mem_write[0]) begin
    if (pe_2_c_mem_q0 < 1) begin
      if (((($bitstoshortreal(pe_2_c_mem_din[0][31:0])) - ($bitstoshortreal(pe_2_c_mem_exp0[pe_2_c_mem_q0][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_2_c_mem_exp0[pe_2_c_mem_q0][31:0])) < 0.0) ? -($bitstoshortreal(pe_2_c_mem_exp0[pe_2_c_mem_q0][31:0])) : ($bitstoshortreal(pe_2_c_mem_exp0[pe_2_c_mem_q0][31:0])))) || ((($bitstoshortreal(pe_2_c_mem_exp0[pe_2_c_mem_q0][31:0])) - ($bitstoshortreal(pe_2_c_mem_din[0][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_2_c_mem_exp0[pe_2_c_mem_q0][31:0])) < 0.0) ? -($bitstoshortreal(pe_2_c_mem_exp0[pe_2_c_mem_q0][31:0])) : ($bitstoshortreal(pe_2_c_mem_exp0[pe_2_c_mem_q0][31:0])))))) begin
        errors = errors + 1;
        $display("MISMATCH pe_2_c_mem[%0d] step %0d: got %h want %h",
                 0, pe_2_c_mem_q0, pe_2_c_mem_din[0], pe_2_c_mem_exp0[pe_2_c_mem_q0]);
      end
      pe_2_c_mem_q0 <= pe_2_c_mem_q0 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_2_c_mem[%0d]", 0);
    end
  end
  reg [31:0] pe_2_c_mem_exp1 [0:0];
  integer pe_2_c_mem_q1 = 0;
  initial begin
    pe_2_c_mem_exp1[0] = 32'h41200000;
  end
  assign pe_2_c_mem_full_n[1] = 1'b1;
  always @(posedge clk) if (rst_n && pe_2_c_mem_write[1]) begin
    if (pe_2_c_mem_q1 < 1) begin
      if (((($bitstoshortreal(pe_2_c_mem_din[1][31:0])) - ($bitstoshortreal(pe_2_c_mem_exp1[pe_2_c_mem_q1][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_2_c_mem_exp1[pe_2_c_mem_q1][31:0])) < 0.0) ? -($bitstoshortreal(pe_2_c_mem_exp1[pe_2_c_mem_q1][31:0])) : ($bitstoshortreal(pe_2_c_mem_exp1[pe_2_c_mem_q1][31:0])))) || ((($bitstoshortreal(pe_2_c_mem_exp1[pe_2_c_mem_q1][31:0])) - ($bitstoshortreal(pe_2_c_mem_din[1][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_2_c_mem_exp1[pe_2_c_mem_q1][31:0])) < 0.0) ? -($bitstoshortreal(pe_2_c_mem_exp1[pe_2_c_mem_q1][31:0])) : ($bitstoshortreal(pe_2_c_mem_exp1[pe_2_c_mem_q1][31:0])))))) begin
        errors = errors + 1;
        $display("MISMATCH pe_2_c_mem[%0d] step %0d: got %h want %h",
                 1, pe_2_c_mem_q1, pe_2_c_mem_din[1], pe_2_c_mem_exp1[pe_2_c_mem_q1]);
      end
      pe_2_c_mem_q1 <= pe_2_c_mem_q1 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_2_c_mem[%0d]", 1);
    end
  end
  reg [31:0] pe_2_c_mem_exp2 [0:0];
  integer pe_2_c_mem_q2 = 0;
  initial begin
    pe_2_c_mem_exp2[0] = 32'h40e00000;
  end
  assign pe_2_c_mem_full_n[2] = 1'b1;
  always @(posedge clk) if (rst_n && pe_2_c_mem_write[2]) begin
    if (pe_2_c_mem_q2 < 1) begin
      if (((($bitstoshortreal(pe_2_c_mem_din[2][31:0])) - ($bitstoshortreal(pe_2_c_mem_exp2[pe_2_c_mem_q2][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_2_c_mem_exp2[pe_2_c_mem_q2][31:0])) < 0.0) ? -($bitstoshortreal(pe_2_c_mem_exp2[pe_2_c_mem_q2][31:0])) : ($bitstoshortreal(pe_2_c_mem_exp2[pe_2_c_mem_q2][31:0])))) || ((($bitstoshortreal(pe_2_c_mem_exp2[pe_2_c_mem_q2][31:0])) - ($bitstoshortreal(pe_2_c_mem_din[2][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_2_c_mem_exp2[pe_2_c_mem_q2][31:0])) < 0.0) ? -($bitstoshortreal(pe_2_c_mem_exp2[pe_2_c_mem_q2][31:0])) : ($bitstoshortreal(pe_2_c_mem_exp2[pe_2_c_mem_q2][31:0])))))) begin
        errors = errors + 1;
        $display("MISMATCH pe_2_c_mem[%0d] step %0d: got %h want %h",
                 2, pe_2_c_mem_q2, pe_2_c_mem_din[2], pe_2_c_mem_exp2[pe_2_c_mem_q2]);
      end
      pe_2_c_mem_q2 <= pe_2_c_mem_q2 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_2_c_mem[%0d]", 2);
    end
  end
  reg [31:0] pe_2_c_mem_exp3 [0:0];
  integer pe_2_c_mem_q3 = 0;
  initial begin
    pe_2_c_mem_exp3[0] = 32'h41300000;
  end
  assign pe_2_c_mem_full_n[3] = 1'b1;
  always @(posedge clk) if (rst_n && pe_2_c_mem_write[3]) begin
    if (pe_2_c_mem_q3 < 1) begin
      if (((($bitstoshortreal(pe_2_c_mem_din[3][31:0])) - ($bitstoshortreal(pe_2_c_mem_exp3[pe_2_c_mem_q3][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_2_c_mem_exp3[pe_2_c_mem_q3][31:0])) < 0.0) ? -($bitstoshortreal(pe_2_c_mem_exp3[pe_2_c_mem_q3][31:0])) : ($bitstoshortreal(pe_2_c_mem_exp3[pe_2_c_mem_q3][31:0])))) || ((($bitstoshortreal(pe_2_c_mem_exp3[pe_2_c_mem_q3][31:0])) - ($bitstoshortreal(pe_2_c_mem_din[3][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_2_c_mem_exp3[pe_2_c_mem_q3][31:0])) < 0.0) ? -($bitstoshortreal(pe_2_c_mem_exp3[pe_2_c_mem_q3][31:0])) : ($bitstoshortreal(pe_2_c_mem_exp3[pe_2_c_mem_q3][31:0])))))) begin
        errors = errors + 1;
        $display("MISMATCH pe_2_c_mem[%0d] step %0d: got %h want %h",
                 3, pe_2_c_mem_q3, pe_2_c_mem_din[3], pe_2_c_mem_exp3[pe_2_c_mem_q3]);
      end
      pe_2_c_mem_q3 <= pe_2_c_mem_q3 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_2_c_mem[%0d]", 3);
    end
  end
  wire [31:0] pe_3_c_mem_din [0:3];
  wire pe_3_c_mem_write [0:3];
  wire pe_3_c_mem_full_n [0:3];
  reg [31:0] pe_3_c_mem_exp0 [0:0];
  integer pe_3_c_mem_q0 = 0;
  initial begin
    pe_3_c_mem_exp0[0] = 32'h41100000;
  end
  assign pe_3_c_mem_full_n[0] = 1'b1;
  always @(posedge clk) if (rst_n && pe_3_c_mem_write[0]) begin
    if (pe_3_c_mem_q0 < 1) begin
      if (((($bitstoshortreal(pe_3_c_mem_din[0][31:0])) - ($bitstoshortreal(pe_3_c_mem_exp0[pe_3_c_mem_q0][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_3_c_mem_exp0[pe_3_c_mem_q0][31:0])) < 0.0) ? -($bitstoshortreal(pe_3_c_mem_exp0[pe_3_c_mem_q0][31:0])) : ($bitstoshortreal(pe_3_c_mem_exp0[pe_3_c_mem_q0][31:0])))) || ((($bitstoshortreal(pe_3_c_mem_exp0[pe_3_c_mem_q0][31:0])) - ($bitstoshortreal(pe_3_c_mem_din[0][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_3_c_mem_exp0[pe_3_c_mem_q0][31:0])) < 0.0) ? -($bitstoshortreal(pe_3_c_mem_exp0[pe_3_c_mem_q0][31:0])) : ($bitstoshortreal(pe_3_c_mem_exp0[pe_3_c_mem_q0][31:0])))))) begin
        errors = errors + 1;
        $display("MISMATCH pe_3_c_mem[%0d] step %0d: got %h want %h",
                 0, pe_3_c_mem_q0, pe_3_c_mem_din[0], pe_3_c_mem_exp0[pe_3_c_mem_q0]);
      end
      pe_3_c_mem_q0 <= pe_3_c_mem_q0 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_3_c_mem[%0d]", 0);
    end
  end
  reg [31:0] pe_3_c_mem_exp1 [0:0];
  integer pe_3_c_mem_q1 = 0;
  initial begin
    pe_3_c_mem_exp1[0] = 32'h00000000;
  end
  assign pe_3_c_mem_full_n[1] = 1'b1;
  always @(posedge clk) if (rst_n && pe_3_c_mem_write[1]) begin
    if (pe_3_c_mem_q1 < 1) begin
      if (((($bitstoshortreal(pe_3_c_mem_din[1][31:0])) - ($bitstoshortreal(pe_3_c_mem_exp1[pe_3_c_mem_q1][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_3_c_mem_exp1[pe_3_c_mem_q1][31:0])) < 0.0) ? -($bitstoshortreal(pe_3_c_mem_exp1[pe_3_c_mem_q1][31:0])) : ($bitstoshortreal(pe_3_c_mem_exp1[pe_3_c_mem_q1][31:0])))) || ((($bitstoshortreal(pe_3_c_mem_exp1[pe_3_c_mem_q1][31:0])) - ($bitstoshortreal(pe_3_c_mem_din[1][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_3_c_mem_exp1[pe_3_c_mem_q1][31:0])) < 0.0) ? -($bitstoshortreal(pe_3_c_mem_exp1[pe_3_c_mem_q1][31:0])) : ($bitstoshortreal(pe_3_c_mem_exp1[pe_3_c_mem_q1][31:0])))))) begin
        errors = errors + 1;
        $display("MISMATCH pe_3_c_mem[%0d] step %0d: got %h want %h",
                 1, pe_3_c_mem_q1, pe_3_c_mem_din[1], pe_3_c_mem_exp1[pe_3_c_mem_q1]);
      end
      pe_3_c_mem_q1 <= pe_3_c_mem_q1 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_3_c_mem[%0d]", 1);
    end
  end
  reg [31:0] pe_3_c_mem_exp2 [0:0];
  integer pe_3_c_mem_q2 = 0;
  initial begin
    pe_3_c_mem_exp2[0] = 32'h41100000;
  end
  assign pe_3_c_mem_full_n[2] = 1'b1;
  always @(posedge clk) if (rst_n && pe_3_c_mem_write[2]) begin
    if (pe_3_c_mem_q2 < 1) begin
      if (((($bitstoshortreal(pe_3_c_mem_din[2][31:0])) - ($bitstoshortreal(pe_3_c_mem_exp2[pe_3_c_mem_q2][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_3_c_mem_exp2[pe_3_c_mem_q2][31:0])) < 0.0) ? -($bitstoshortreal(pe_3_c_mem_exp2[pe_3_c_mem_q2][31:0])) : ($bitstoshortreal(pe_3_c_mem_exp2[pe_3_c_mem_q2][31:0])))) || ((($bitstoshortreal(pe_3_c_mem_exp2[pe_3_c_mem_q2][31:0])) - ($bitstoshortreal(pe_3_c_mem_din[2][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_3_c_mem_exp2[pe_3_c_mem_q2][31:0])) < 0.0) ? -($bitstoshortreal(pe_3_c_mem_exp2[pe_3_c_mem_q2][31:0])) : ($bitstoshortreal(pe_3_c_mem_exp2[pe_3_c_mem_q2][31:0])))))) begin
        errors = errors + 1;
        $display("MISMATCH pe_3_c_mem[%0d] step %0d: got %h want %h",
                 2, pe_3_c_mem_q2, pe_3_c_mem_din[2], pe_3_c_mem_exp2[pe_3_c_mem_q2]);
      end
      pe_3_c_mem_q2 <= pe_3_c_mem_q2 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_3_c_mem[%0d]", 2);
    end
  end
  reg [31:0] pe_3_c_mem_exp3 [0:0];
  integer pe_3_c_mem_q3 = 0;
  initial begin
    pe_3_c_mem_exp3[0] = 32'h40000000;
  end
  assign pe_3_c_mem_full_n[3] = 1'b1;
  always @(posedge clk) if (rst_n && pe_3_c_mem_write[3]) begin
    if (pe_3_c_mem_q3 < 1) begin
      if (((($bitstoshortreal(pe_3_c_mem_din[3][31:0])) - ($bitstoshortreal(pe_3_c_mem_exp3[pe_3_c_mem_q3][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_3_c_mem_exp3[pe_3_c_mem_q3][31:0])) < 0.0) ? -($bitstoshortreal(pe_3_c_mem_exp3[pe_3_c_mem_q3][31:0])) : ($bitstoshortreal(pe_3_c_mem_exp3[pe_3_c_mem_q3][31:0])))) || ((($bitstoshortreal(pe_3_c_mem_exp3[pe_3_c_mem_q3][31:0])) - ($bitstoshortreal(pe_3_c_mem_din[3][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(pe_3_c_mem_exp3[pe_3_c_mem_q3][31:0])) < 0.0) ? -($bitstoshortreal(pe_3_c_mem_exp3[pe_3_c_mem_q3][31:0])) : ($bitstoshortreal(pe_3_c_mem_exp3[pe_3_c_mem_q3][31:0])))))) begin
        errors = errors + 1;
        $display("MISMATCH pe_3_c_mem[%0d] step %0d: got %h want %h",
                 3, pe_3_c_mem_q3, pe_3_c_mem_din[3], pe_3_c_mem_exp3[pe_3_c_mem_q3]);
      end
      pe_3_c_mem_q3 <= pe_3_c_mem_q3 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_3_c_mem[%0d]", 3);
    end
  end
  spmw_top dut (.ap_clk(clk), .ap_rst_n(rst_n), .pe_west_bind_dout(pe_west_bind_dout), .pe_west_bind_empty_n(pe_west_bind_empty_n), .pe_west_bind_read(pe_west_bind_read), .pe_north_bind_dout(pe_north_bind_dout), .pe_north_bind_empty_n(pe_north_bind_empty_n), .pe_north_bind_read(pe_north_bind_read), .pe_1_west_bind_dout(pe_1_west_bind_dout), .pe_1_west_bind_empty_n(pe_1_west_bind_empty_n), .pe_1_west_bind_read(pe_1_west_bind_read), .pe_1_north_bind_dout(pe_1_north_bind_dout), .pe_1_north_bind_empty_n(pe_1_north_bind_empty_n), .pe_1_north_bind_read(pe_1_north_bind_read), .pe_2_west_bind_dout(pe_2_west_bind_dout), .pe_2_west_bind_empty_n(pe_2_west_bind_empty_n), .pe_2_west_bind_read(pe_2_west_bind_read), .pe_2_north_bind_dout(pe_2_north_bind_dout), .pe_2_north_bind_empty_n(pe_2_north_bind_empty_n), .pe_2_north_bind_read(pe_2_north_bind_read), .pe_3_west_bind_dout(pe_3_west_bind_dout), .pe_3_west_bind_empty_n(pe_3_west_bind_empty_n), .pe_3_west_bind_read(pe_3_west_bind_read), .pe_3_north_bind_dout(pe_3_north_bind_dout), .pe_3_north_bind_empty_n(pe_3_north_bind_empty_n), .pe_3_north_bind_read(pe_3_north_bind_read), .pe_c_mem_din(pe_c_mem_din), .pe_c_mem_write(pe_c_mem_write), .pe_c_mem_full_n(pe_c_mem_full_n), .pe_1_c_mem_din(pe_1_c_mem_din), .pe_1_c_mem_write(pe_1_c_mem_write), .pe_1_c_mem_full_n(pe_1_c_mem_full_n), .pe_2_c_mem_din(pe_2_c_mem_din), .pe_2_c_mem_write(pe_2_c_mem_write), .pe_2_c_mem_full_n(pe_2_c_mem_full_n), .pe_3_c_mem_din(pe_3_c_mem_din), .pe_3_c_mem_write(pe_3_c_mem_write), .pe_3_c_mem_full_n(pe_3_c_mem_full_n));
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
