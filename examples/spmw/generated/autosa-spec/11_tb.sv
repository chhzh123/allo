`timescale 1ns/1ps

module tb;
  reg clk = 0, rst_n = 0;
  always #5 clk = ~clk;
  integer errors = 0;
  integer produced = 0;
  integer first = -1;
  localparam integer TOTAL = 16;
  wire [31:0] feed_up_bind_dout [0:0];
  wire feed_up_bind_empty_n [0:0];
  wire feed_up_bind_read [0:0];
  reg [31:0] feed_up_bind_src0 [0:3];
  integer feed_up_bind_p0 = 0;
  initial begin
    feed_up_bind_src0[0] = 32'h00010102;
    feed_up_bind_src0[1] = 32'h00000000;
    feed_up_bind_src0[2] = 32'h02010200;
    feed_up_bind_src0[3] = 32'h02020101;
  end
  assign feed_up_bind_dout[0] = feed_up_bind_src0[feed_up_bind_p0 < 4 ? feed_up_bind_p0 : 3];
  assign feed_up_bind_empty_n[0] = (feed_up_bind_p0 < 4);
  always @(posedge clk) if (rst_n && feed_up_bind_read[0] && feed_up_bind_empty_n[0]) feed_up_bind_p0 <= feed_up_bind_p0 + 1;
  wire [31:0] feed_2_up_bind_dout [0:0];
  wire feed_2_up_bind_empty_n [0:0];
  wire feed_2_up_bind_read [0:0];
  reg [31:0] feed_2_up_bind_src0 [0:3];
  integer feed_2_up_bind_p0 = 0;
  initial begin
    feed_2_up_bind_src0[0] = 32'h02010101;
    feed_2_up_bind_src0[1] = 32'h00020200;
    feed_2_up_bind_src0[2] = 32'h00010201;
    feed_2_up_bind_src0[3] = 32'h00020202;
  end
  assign feed_2_up_bind_dout[0] = feed_2_up_bind_src0[feed_2_up_bind_p0 < 4 ? feed_2_up_bind_p0 : 3];
  assign feed_2_up_bind_empty_n[0] = (feed_2_up_bind_p0 < 4);
  always @(posedge clk) if (rst_n && feed_2_up_bind_read[0] && feed_2_up_bind_empty_n[0]) feed_2_up_bind_p0 <= feed_2_up_bind_p0 + 1;
  wire [31:0] pe_c_out_bind_din [0:3];
  wire pe_c_out_bind_write [0:3];
  wire pe_c_out_bind_full_n [0:3];
  reg [31:0] pe_c_out_bind_exp0 [0:3];
  integer pe_c_out_bind_q0 = 0;
  initial begin
    pe_c_out_bind_exp0[0] = 32'h00000006;
    pe_c_out_bind_exp0[1] = 32'h00000006;
    pe_c_out_bind_exp0[2] = 32'h00000005;
    pe_c_out_bind_exp0[3] = 32'h00000004;
  end
  assign pe_c_out_bind_full_n[0] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_out_bind_write[0]) begin
    if (pe_c_out_bind_q0 < 4) begin
      if (pe_c_out_bind_din[0] !== pe_c_out_bind_exp0[pe_c_out_bind_q0]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_out_bind[%0d] step %0d: got %h want %h",
                 0, pe_c_out_bind_q0, pe_c_out_bind_din[0], pe_c_out_bind_exp0[pe_c_out_bind_q0]);
      end
      pe_c_out_bind_q0 <= pe_c_out_bind_q0 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_out_bind[%0d]", 0);
    end
  end
  reg [31:0] pe_c_out_bind_exp1 [0:3];
  integer pe_c_out_bind_q1 = 0;
  initial begin
    pe_c_out_bind_exp1[0] = 32'h00000008;
    pe_c_out_bind_exp1[1] = 32'h00000007;
    pe_c_out_bind_exp1[2] = 32'h00000007;
    pe_c_out_bind_exp1[3] = 32'h00000004;
  end
  assign pe_c_out_bind_full_n[1] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_out_bind_write[1]) begin
    if (pe_c_out_bind_q1 < 4) begin
      if (pe_c_out_bind_din[1] !== pe_c_out_bind_exp1[pe_c_out_bind_q1]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_out_bind[%0d] step %0d: got %h want %h",
                 1, pe_c_out_bind_q1, pe_c_out_bind_din[1], pe_c_out_bind_exp1[pe_c_out_bind_q1]);
      end
      pe_c_out_bind_q1 <= pe_c_out_bind_q1 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_out_bind[%0d]", 1);
    end
  end
  reg [31:0] pe_c_out_bind_exp2 [0:3];
  integer pe_c_out_bind_q2 = 0;
  initial begin
    pe_c_out_bind_exp2[0] = 32'h00000006;
    pe_c_out_bind_exp2[1] = 32'h00000006;
    pe_c_out_bind_exp2[2] = 32'h00000005;
    pe_c_out_bind_exp2[3] = 32'h00000004;
  end
  assign pe_c_out_bind_full_n[2] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_out_bind_write[2]) begin
    if (pe_c_out_bind_q2 < 4) begin
      if (pe_c_out_bind_din[2] !== pe_c_out_bind_exp2[pe_c_out_bind_q2]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_out_bind[%0d] step %0d: got %h want %h",
                 2, pe_c_out_bind_q2, pe_c_out_bind_din[2], pe_c_out_bind_exp2[pe_c_out_bind_q2]);
      end
      pe_c_out_bind_q2 <= pe_c_out_bind_q2 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_out_bind[%0d]", 2);
    end
  end
  reg [31:0] pe_c_out_bind_exp3 [0:3];
  integer pe_c_out_bind_q3 = 0;
  initial begin
    pe_c_out_bind_exp3[0] = 32'h00000000;
    pe_c_out_bind_exp3[1] = 32'h00000002;
    pe_c_out_bind_exp3[2] = 32'h00000002;
    pe_c_out_bind_exp3[3] = 32'h00000004;
  end
  assign pe_c_out_bind_full_n[3] = 1'b1;
  always @(posedge clk) if (rst_n && pe_c_out_bind_write[3]) begin
    if (pe_c_out_bind_q3 < 4) begin
      if (pe_c_out_bind_din[3] !== pe_c_out_bind_exp3[pe_c_out_bind_q3]) begin
        errors = errors + 1;
        $display("MISMATCH pe_c_out_bind[%0d] step %0d: got %h want %h",
                 3, pe_c_out_bind_q3, pe_c_out_bind_din[3], pe_c_out_bind_exp3[pe_c_out_bind_q3]);
      end
      pe_c_out_bind_q3 <= pe_c_out_bind_q3 + 1;
      produced = produced + 1;
    end else begin
      errors = errors + 1;
      $display("EXTRA TOKEN on pe_c_out_bind[%0d]", 3);
    end
  end
  spmw_top dut (.ap_clk(clk), .ap_rst_n(rst_n), .feed_up_bind_dout(feed_up_bind_dout), .feed_up_bind_empty_n(feed_up_bind_empty_n), .feed_up_bind_read(feed_up_bind_read), .feed_2_up_bind_dout(feed_2_up_bind_dout), .feed_2_up_bind_empty_n(feed_2_up_bind_empty_n), .feed_2_up_bind_read(feed_2_up_bind_read), .pe_c_out_bind_din(pe_c_out_bind_din), .pe_c_out_bind_write(pe_c_out_bind_write), .pe_c_out_bind_full_n(pe_c_out_bind_full_n));
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
