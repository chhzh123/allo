`timescale 1ns/1ps

module tb;
  reg clk = 0, rst_n = 0;
  always #5 clk = ~clk;
  integer errors = 0;
  integer produced = 0;
  localparam integer TOTAL = 8;
  wire [63:0] bfly_up_in_bind_dout [0:3];
  wire bfly_up_in_bind_empty_n [0:3];
  wire bfly_up_in_bind_read [0:3];
  reg [63:0] bfly_up_in_bind_src0 [0:0];
  integer bfly_up_in_bind_p0 = 0;
  initial begin
    bfly_up_in_bind_src0[0] = 64'h3f80000040000000;
  end
  assign bfly_up_in_bind_dout[0] = bfly_up_in_bind_src0[bfly_up_in_bind_p0 < 1 ? bfly_up_in_bind_p0 : 0];
  assign bfly_up_in_bind_empty_n[0] = (bfly_up_in_bind_p0 < 1);
  always @(posedge clk) if (rst_n && bfly_up_in_bind_read[0] && bfly_up_in_bind_empty_n[0]) bfly_up_in_bind_p0 <= bfly_up_in_bind_p0 + 1;
  reg [63:0] bfly_up_in_bind_src1 [0:0];
  integer bfly_up_in_bind_p1 = 0;
  initial begin
    bfly_up_in_bind_src1[0] = 64'h0000000000000000;
  end
  assign bfly_up_in_bind_dout[1] = bfly_up_in_bind_src1[bfly_up_in_bind_p1 < 1 ? bfly_up_in_bind_p1 : 0];
  assign bfly_up_in_bind_empty_n[1] = (bfly_up_in_bind_p1 < 1);
  always @(posedge clk) if (rst_n && bfly_up_in_bind_read[1] && bfly_up_in_bind_empty_n[1]) bfly_up_in_bind_p1 <= bfly_up_in_bind_p1 + 1;
  reg [63:0] bfly_up_in_bind_src2 [0:0];
  integer bfly_up_in_bind_p2 = 0;
  initial begin
    bfly_up_in_bind_src2[0] = 64'h000000003f800000;
  end
  assign bfly_up_in_bind_dout[2] = bfly_up_in_bind_src2[bfly_up_in_bind_p2 < 1 ? bfly_up_in_bind_p2 : 0];
  assign bfly_up_in_bind_empty_n[2] = (bfly_up_in_bind_p2 < 1);
  always @(posedge clk) if (rst_n && bfly_up_in_bind_read[2] && bfly_up_in_bind_empty_n[2]) bfly_up_in_bind_p2 <= bfly_up_in_bind_p2 + 1;
  reg [63:0] bfly_up_in_bind_src3 [0:0];
  integer bfly_up_in_bind_p3 = 0;
  initial begin
    bfly_up_in_bind_src3[0] = 64'h0000000000000000;
  end
  assign bfly_up_in_bind_dout[3] = bfly_up_in_bind_src3[bfly_up_in_bind_p3 < 1 ? bfly_up_in_bind_p3 : 0];
  assign bfly_up_in_bind_empty_n[3] = (bfly_up_in_bind_p3 < 1);
  always @(posedge clk) if (rst_n && bfly_up_in_bind_read[3] && bfly_up_in_bind_empty_n[3]) bfly_up_in_bind_p3 <= bfly_up_in_bind_p3 + 1;
  wire [63:0] bfly_lo_in_bind_dout [0:3];
  wire bfly_lo_in_bind_empty_n [0:3];
  wire bfly_lo_in_bind_read [0:3];
  reg [63:0] bfly_lo_in_bind_src0 [0:0];
  integer bfly_lo_in_bind_p0 = 0;
  initial begin
    bfly_lo_in_bind_src0[0] = 64'h4000000000000000;
  end
  assign bfly_lo_in_bind_dout[0] = bfly_lo_in_bind_src0[bfly_lo_in_bind_p0 < 1 ? bfly_lo_in_bind_p0 : 0];
  assign bfly_lo_in_bind_empty_n[0] = (bfly_lo_in_bind_p0 < 1);
  always @(posedge clk) if (rst_n && bfly_lo_in_bind_read[0] && bfly_lo_in_bind_empty_n[0]) bfly_lo_in_bind_p0 <= bfly_lo_in_bind_p0 + 1;
  reg [63:0] bfly_lo_in_bind_src1 [0:0];
  integer bfly_lo_in_bind_p1 = 0;
  initial begin
    bfly_lo_in_bind_src1[0] = 64'h3f8000003f800000;
  end
  assign bfly_lo_in_bind_dout[1] = bfly_lo_in_bind_src1[bfly_lo_in_bind_p1 < 1 ? bfly_lo_in_bind_p1 : 0];
  assign bfly_lo_in_bind_empty_n[1] = (bfly_lo_in_bind_p1 < 1);
  always @(posedge clk) if (rst_n && bfly_lo_in_bind_read[1] && bfly_lo_in_bind_empty_n[1]) bfly_lo_in_bind_p1 <= bfly_lo_in_bind_p1 + 1;
  reg [63:0] bfly_lo_in_bind_src2 [0:0];
  integer bfly_lo_in_bind_p2 = 0;
  initial begin
    bfly_lo_in_bind_src2[0] = 64'h400000003f800000;
  end
  assign bfly_lo_in_bind_dout[2] = bfly_lo_in_bind_src2[bfly_lo_in_bind_p2 < 1 ? bfly_lo_in_bind_p2 : 0];
  assign bfly_lo_in_bind_empty_n[2] = (bfly_lo_in_bind_p2 < 1);
  always @(posedge clk) if (rst_n && bfly_lo_in_bind_read[2] && bfly_lo_in_bind_empty_n[2]) bfly_lo_in_bind_p2 <= bfly_lo_in_bind_p2 + 1;
  reg [63:0] bfly_lo_in_bind_src3 [0:0];
  integer bfly_lo_in_bind_p3 = 0;
  initial begin
    bfly_lo_in_bind_src3[0] = 64'h4000000040000000;
  end
  assign bfly_lo_in_bind_dout[3] = bfly_lo_in_bind_src3[bfly_lo_in_bind_p3 < 1 ? bfly_lo_in_bind_p3 : 0];
  assign bfly_lo_in_bind_empty_n[3] = (bfly_lo_in_bind_p3 < 1);
  always @(posedge clk) if (rst_n && bfly_lo_in_bind_read[3] && bfly_lo_in_bind_empty_n[3]) bfly_lo_in_bind_p3 <= bfly_lo_in_bind_p3 + 1;
  wire [63:0] bfly_up_out_bind_din [0:3];
  wire bfly_up_out_bind_write [0:3];
  wire bfly_up_out_bind_full_n [0:3];
  reg [63:0] bfly_up_out_bind_exp0 [0:0];
  integer bfly_up_out_bind_q0 = 0;
  initial begin
    bfly_up_out_bind_exp0[0] = 64'h4100000040e00000;
  end
  assign bfly_up_out_bind_full_n[0] = 1'b1;
  always @(posedge clk) if (rst_n && bfly_up_out_bind_write[0]) begin
    if (bfly_up_out_bind_q0 < 1) begin
      if (((($bitstoshortreal(bfly_up_out_bind_din[0][31:0])) - ($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][31:0])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][31:0])) : ($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][31:0])))) || ((($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][31:0])) - ($bitstoshortreal(bfly_up_out_bind_din[0][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][31:0])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][31:0])) : ($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][31:0]))))) || ((($bitstoshortreal(bfly_up_out_bind_din[0][63:32])) - ($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][63:32])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][63:32])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][63:32])) : ($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][63:32])))) || ((($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][63:32])) - ($bitstoshortreal(bfly_up_out_bind_din[0][63:32]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][63:32])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][63:32])) : ($bitstoshortreal(bfly_up_out_bind_exp0[bfly_up_out_bind_q0][63:32])))))) begin
        errors = errors + 1;
        $display("MISMATCH bfly_up_out_bind[%0d] step %0d: got %h want %h",
                 0, bfly_up_out_bind_q0, bfly_up_out_bind_din[0], bfly_up_out_bind_exp0[bfly_up_out_bind_q0]);
      end
      bfly_up_out_bind_q0 <= bfly_up_out_bind_q0 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on bfly_up_out_bind[%0d]", 0);
    end
  end
  reg [63:0] bfly_up_out_bind_exp1 [0:0];
  integer bfly_up_out_bind_q1 = 0;
  initial begin
    bfly_up_out_bind_exp1[0] = 64'h3fb504f3bed413cc;
  end
  assign bfly_up_out_bind_full_n[1] = 1'b1;
  always @(posedge clk) if (rst_n && bfly_up_out_bind_write[1]) begin
    if (bfly_up_out_bind_q1 < 1) begin
      if (((($bitstoshortreal(bfly_up_out_bind_din[1][31:0])) - ($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][31:0])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][31:0])) : ($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][31:0])))) || ((($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][31:0])) - ($bitstoshortreal(bfly_up_out_bind_din[1][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][31:0])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][31:0])) : ($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][31:0]))))) || ((($bitstoshortreal(bfly_up_out_bind_din[1][63:32])) - ($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][63:32])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][63:32])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][63:32])) : ($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][63:32])))) || ((($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][63:32])) - ($bitstoshortreal(bfly_up_out_bind_din[1][63:32]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][63:32])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][63:32])) : ($bitstoshortreal(bfly_up_out_bind_exp1[bfly_up_out_bind_q1][63:32])))))) begin
        errors = errors + 1;
        $display("MISMATCH bfly_up_out_bind[%0d] step %0d: got %h want %h",
                 1, bfly_up_out_bind_q1, bfly_up_out_bind_din[1], bfly_up_out_bind_exp1[bfly_up_out_bind_q1]);
      end
      bfly_up_out_bind_q1 <= bfly_up_out_bind_q1 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on bfly_up_out_bind[%0d]", 1);
    end
  end
  reg [63:0] bfly_up_out_bind_exp2 [0:0];
  integer bfly_up_out_bind_q2 = 0;
  initial begin
    bfly_up_out_bind_exp2[0] = 64'h400000003f800000;
  end
  assign bfly_up_out_bind_full_n[2] = 1'b1;
  always @(posedge clk) if (rst_n && bfly_up_out_bind_write[2]) begin
    if (bfly_up_out_bind_q2 < 1) begin
      if (((($bitstoshortreal(bfly_up_out_bind_din[2][31:0])) - ($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][31:0])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][31:0])) : ($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][31:0])))) || ((($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][31:0])) - ($bitstoshortreal(bfly_up_out_bind_din[2][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][31:0])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][31:0])) : ($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][31:0]))))) || ((($bitstoshortreal(bfly_up_out_bind_din[2][63:32])) - ($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][63:32])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][63:32])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][63:32])) : ($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][63:32])))) || ((($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][63:32])) - ($bitstoshortreal(bfly_up_out_bind_din[2][63:32]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][63:32])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][63:32])) : ($bitstoshortreal(bfly_up_out_bind_exp2[bfly_up_out_bind_q2][63:32])))))) begin
        errors = errors + 1;
        $display("MISMATCH bfly_up_out_bind[%0d] step %0d: got %h want %h",
                 2, bfly_up_out_bind_q2, bfly_up_out_bind_din[2], bfly_up_out_bind_exp2[bfly_up_out_bind_q2]);
      end
      bfly_up_out_bind_q2 <= bfly_up_out_bind_q2 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on bfly_up_out_bind[%0d]", 2);
    end
  end
  reg [63:0] bfly_up_out_bind_exp3 [0:0];
  integer bfly_up_out_bind_q3 = 0;
  initial begin
    bfly_up_out_bind_exp3[0] = 64'hbf15f61abf9f0ed8;
  end
  assign bfly_up_out_bind_full_n[3] = 1'b1;
  always @(posedge clk) if (rst_n && bfly_up_out_bind_write[3]) begin
    if (bfly_up_out_bind_q3 < 1) begin
      if (((($bitstoshortreal(bfly_up_out_bind_din[3][31:0])) - ($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][31:0])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][31:0])) : ($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][31:0])))) || ((($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][31:0])) - ($bitstoshortreal(bfly_up_out_bind_din[3][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][31:0])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][31:0])) : ($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][31:0]))))) || ((($bitstoshortreal(bfly_up_out_bind_din[3][63:32])) - ($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][63:32])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][63:32])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][63:32])) : ($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][63:32])))) || ((($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][63:32])) - ($bitstoshortreal(bfly_up_out_bind_din[3][63:32]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][63:32])) < 0.0) ? -($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][63:32])) : ($bitstoshortreal(bfly_up_out_bind_exp3[bfly_up_out_bind_q3][63:32])))))) begin
        errors = errors + 1;
        $display("MISMATCH bfly_up_out_bind[%0d] step %0d: got %h want %h",
                 3, bfly_up_out_bind_q3, bfly_up_out_bind_din[3], bfly_up_out_bind_exp3[bfly_up_out_bind_q3]);
      end
      bfly_up_out_bind_q3 <= bfly_up_out_bind_q3 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on bfly_up_out_bind[%0d]", 3);
    end
  end
  wire [63:0] bfly_lo_out_bind_din [0:3];
  wire bfly_lo_out_bind_write [0:3];
  wire bfly_lo_out_bind_full_n [0:3];
  reg [63:0] bfly_lo_out_bind_exp0 [0:0];
  integer bfly_lo_out_bind_q0 = 0;
  initial begin
    bfly_lo_out_bind_exp0[0] = 64'h00000000bf800000;
  end
  assign bfly_lo_out_bind_full_n[0] = 1'b1;
  always @(posedge clk) if (rst_n && bfly_lo_out_bind_write[0]) begin
    if (bfly_lo_out_bind_q0 < 1) begin
      if (((($bitstoshortreal(bfly_lo_out_bind_din[0][31:0])) - ($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][31:0])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][31:0])) : ($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][31:0])))) || ((($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][31:0])) - ($bitstoshortreal(bfly_lo_out_bind_din[0][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][31:0])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][31:0])) : ($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][31:0]))))) || ((($bitstoshortreal(bfly_lo_out_bind_din[0][63:32])) - ($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][63:32])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][63:32])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][63:32])) : ($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][63:32])))) || ((($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][63:32])) - ($bitstoshortreal(bfly_lo_out_bind_din[0][63:32]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][63:32])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][63:32])) : ($bitstoshortreal(bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0][63:32])))))) begin
        errors = errors + 1;
        $display("MISMATCH bfly_lo_out_bind[%0d] step %0d: got %h want %h",
                 0, bfly_lo_out_bind_q0, bfly_lo_out_bind_din[0], bfly_lo_out_bind_exp0[bfly_lo_out_bind_q0]);
      end
      bfly_lo_out_bind_q0 <= bfly_lo_out_bind_q0 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on bfly_lo_out_bind[%0d]", 0);
    end
  end
  reg [63:0] bfly_lo_out_bind_exp1 [0:0];
  integer bfly_lo_out_bind_q1 = 0;
  initial begin
    bfly_lo_out_bind_exp1[0] = 64'hbfb504f3401a827a;
  end
  assign bfly_lo_out_bind_full_n[1] = 1'b1;
  always @(posedge clk) if (rst_n && bfly_lo_out_bind_write[1]) begin
    if (bfly_lo_out_bind_q1 < 1) begin
      if (((($bitstoshortreal(bfly_lo_out_bind_din[1][31:0])) - ($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][31:0])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][31:0])) : ($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][31:0])))) || ((($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][31:0])) - ($bitstoshortreal(bfly_lo_out_bind_din[1][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][31:0])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][31:0])) : ($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][31:0]))))) || ((($bitstoshortreal(bfly_lo_out_bind_din[1][63:32])) - ($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][63:32])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][63:32])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][63:32])) : ($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][63:32])))) || ((($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][63:32])) - ($bitstoshortreal(bfly_lo_out_bind_din[1][63:32]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][63:32])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][63:32])) : ($bitstoshortreal(bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1][63:32])))))) begin
        errors = errors + 1;
        $display("MISMATCH bfly_lo_out_bind[%0d] step %0d: got %h want %h",
                 1, bfly_lo_out_bind_q1, bfly_lo_out_bind_din[1], bfly_lo_out_bind_exp1[bfly_lo_out_bind_q1]);
      end
      bfly_lo_out_bind_q1 <= bfly_lo_out_bind_q1 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on bfly_lo_out_bind[%0d]", 1);
    end
  end
  reg [63:0] bfly_lo_out_bind_exp2 [0:0];
  integer bfly_lo_out_bind_q2 = 0;
  initial begin
    bfly_lo_out_bind_exp2[0] = 64'h400000003f800000;
  end
  assign bfly_lo_out_bind_full_n[2] = 1'b1;
  always @(posedge clk) if (rst_n && bfly_lo_out_bind_write[2]) begin
    if (bfly_lo_out_bind_q2 < 1) begin
      if (((($bitstoshortreal(bfly_lo_out_bind_din[2][31:0])) - ($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][31:0])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][31:0])) : ($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][31:0])))) || ((($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][31:0])) - ($bitstoshortreal(bfly_lo_out_bind_din[2][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][31:0])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][31:0])) : ($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][31:0]))))) || ((($bitstoshortreal(bfly_lo_out_bind_din[2][63:32])) - ($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][63:32])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][63:32])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][63:32])) : ($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][63:32])))) || ((($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][63:32])) - ($bitstoshortreal(bfly_lo_out_bind_din[2][63:32]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][63:32])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][63:32])) : ($bitstoshortreal(bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2][63:32])))))) begin
        errors = errors + 1;
        $display("MISMATCH bfly_lo_out_bind[%0d] step %0d: got %h want %h",
                 2, bfly_lo_out_bind_q2, bfly_lo_out_bind_din[2], bfly_lo_out_bind_exp2[bfly_lo_out_bind_q2]);
      end
      bfly_lo_out_bind_q2 <= bfly_lo_out_bind_q2 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on bfly_lo_out_bind[%0d]", 2);
    end
  end
  reg [63:0] bfly_lo_out_bind_exp3 [0:0];
  integer bfly_lo_out_bind_q3 = 0;
  initial begin
    bfly_lo_out_bind_exp3[0] = 64'hc05a827a40e7c3b6;
  end
  assign bfly_lo_out_bind_full_n[3] = 1'b1;
  always @(posedge clk) if (rst_n && bfly_lo_out_bind_write[3]) begin
    if (bfly_lo_out_bind_q3 < 1) begin
      if (((($bitstoshortreal(bfly_lo_out_bind_din[3][31:0])) - ($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][31:0])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][31:0])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][31:0])) : ($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][31:0])))) || ((($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][31:0])) - ($bitstoshortreal(bfly_lo_out_bind_din[3][31:0]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][31:0])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][31:0])) : ($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][31:0]))))) || ((($bitstoshortreal(bfly_lo_out_bind_din[3][63:32])) - ($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][63:32])) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][63:32])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][63:32])) : ($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][63:32])))) || ((($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][63:32])) - ($bitstoshortreal(bfly_lo_out_bind_din[3][63:32]))) > 1e-06 + 1e-05 * ((($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][63:32])) < 0.0) ? -($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][63:32])) : ($bitstoshortreal(bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3][63:32])))))) begin
        errors = errors + 1;
        $display("MISMATCH bfly_lo_out_bind[%0d] step %0d: got %h want %h",
                 3, bfly_lo_out_bind_q3, bfly_lo_out_bind_din[3], bfly_lo_out_bind_exp3[bfly_lo_out_bind_q3]);
      end
      bfly_lo_out_bind_q3 <= bfly_lo_out_bind_q3 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on bfly_lo_out_bind[%0d]", 3);
    end
  end
  spmw_top dut (.ap_clk(clk), .ap_rst_n(rst_n), .bfly_up_in_bind_dout(bfly_up_in_bind_dout), .bfly_up_in_bind_empty_n(bfly_up_in_bind_empty_n), .bfly_up_in_bind_read(bfly_up_in_bind_read), .bfly_lo_in_bind_dout(bfly_lo_in_bind_dout), .bfly_lo_in_bind_empty_n(bfly_lo_in_bind_empty_n), .bfly_lo_in_bind_read(bfly_lo_in_bind_read), .bfly_up_out_bind_din(bfly_up_out_bind_din), .bfly_up_out_bind_write(bfly_up_out_bind_write), .bfly_up_out_bind_full_n(bfly_up_out_bind_full_n), .bfly_lo_out_bind_din(bfly_lo_out_bind_din), .bfly_lo_out_bind_write(bfly_lo_out_bind_write), .bfly_lo_out_bind_full_n(bfly_lo_out_bind_full_n));
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
