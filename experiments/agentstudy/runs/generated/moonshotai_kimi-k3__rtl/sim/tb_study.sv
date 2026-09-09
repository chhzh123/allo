// One harness for every representation. Whatever the design was written in, it
// is elaborated as `dut_norm` with this port list, driven with the same
// vectors, and graded on the same two numbers: mismatching values, and the
// cycles from a product's first input transfer to its last output transfer.
`timescale 1ns/1ps

module tb;
  parameter integer NPROD = 2;          // products in the vector file
  parameter integer STEPS = 8;          // values per port per product
  localparam integer PER  = STEPS * 8;  // values per port-set per product

  reg ap_clk = 0, ap_rst_n = 0;
  always #2 ap_clk = ~ap_clk;           // 250 MHz simulation clock

  // ---- vectors -------------------------------------------------------------
  reg  [7:0]  a_mem [0:NPROD*PER-1];
  reg  [7:0]  b_mem [0:NPROD*PER-1];
  reg  [31:0] c_mem [0:NPROD*PER-1];
  reg [1023:0] vecdir;

  // ---- dut wires -----------------------------------------------------------
  wire [7:0]  a_dout  [0:7];  wire a_empty_n [0:7];  wire a_read  [0:7];
  wire [7:0]  b_dout  [0:7];  wire b_empty_n [0:7];  wire b_read  [0:7];
  wire [31:0] c_din   [0:7];  wire c_full_n  [0:7];  wire c_write [0:7];

  // ---- per-port progress ---------------------------------------------------
  integer a_cnt [0:7];  integer b_cnt [0:7];  integer c_cnt [0:7];
  integer cyc, bad, done_ports, i, j, p;
  integer first_in  [0:15];   // cycle of each product's first input transfer
  integer last_out  [0:15];   // cycle of each product's last output transfer
  reg     started   [0:15];

  // The harness always offers input and always accepts output, so the design
  // sets the pace and the measured cycles are the design's own.
  genvar g;
  generate
    for (g = 0; g < 8; g = g + 1) begin : feed
      assign a_dout[g]   = a_mem[(a_cnt[g] / STEPS) * PER + g * STEPS + (a_cnt[g] % STEPS)];
      assign b_dout[g]   = b_mem[(b_cnt[g] / STEPS) * PER + g * STEPS + (b_cnt[g] % STEPS)];
      assign a_empty_n[g] = (a_cnt[g] < NPROD * STEPS);
      assign b_empty_n[g] = (b_cnt[g] < NPROD * STEPS);
      assign c_full_n[g]  = 1'b1;
    end
  endgenerate

  dut_norm dut (
    .ap_clk(ap_clk), .ap_rst_n(ap_rst_n),
    .a_in_0_dout(a_dout[0]), .a_in_0_empty_n(a_empty_n[0]), .a_in_0_read(a_read[0]),
    .a_in_1_dout(a_dout[1]), .a_in_1_empty_n(a_empty_n[1]), .a_in_1_read(a_read[1]),
    .a_in_2_dout(a_dout[2]), .a_in_2_empty_n(a_empty_n[2]), .a_in_2_read(a_read[2]),
    .a_in_3_dout(a_dout[3]), .a_in_3_empty_n(a_empty_n[3]), .a_in_3_read(a_read[3]),
    .a_in_4_dout(a_dout[4]), .a_in_4_empty_n(a_empty_n[4]), .a_in_4_read(a_read[4]),
    .a_in_5_dout(a_dout[5]), .a_in_5_empty_n(a_empty_n[5]), .a_in_5_read(a_read[5]),
    .a_in_6_dout(a_dout[6]), .a_in_6_empty_n(a_empty_n[6]), .a_in_6_read(a_read[6]),
    .a_in_7_dout(a_dout[7]), .a_in_7_empty_n(a_empty_n[7]), .a_in_7_read(a_read[7]),
    .b_in_0_dout(b_dout[0]), .b_in_0_empty_n(b_empty_n[0]), .b_in_0_read(b_read[0]),
    .b_in_1_dout(b_dout[1]), .b_in_1_empty_n(b_empty_n[1]), .b_in_1_read(b_read[1]),
    .b_in_2_dout(b_dout[2]), .b_in_2_empty_n(b_empty_n[2]), .b_in_2_read(b_read[2]),
    .b_in_3_dout(b_dout[3]), .b_in_3_empty_n(b_empty_n[3]), .b_in_3_read(b_read[3]),
    .b_in_4_dout(b_dout[4]), .b_in_4_empty_n(b_empty_n[4]), .b_in_4_read(b_read[4]),
    .b_in_5_dout(b_dout[5]), .b_in_5_empty_n(b_empty_n[5]), .b_in_5_read(b_read[5]),
    .b_in_6_dout(b_dout[6]), .b_in_6_empty_n(b_empty_n[6]), .b_in_6_read(b_read[6]),
    .b_in_7_dout(b_dout[7]), .b_in_7_empty_n(b_empty_n[7]), .b_in_7_read(b_read[7]),
    .c_out_0_din(c_din[0]), .c_out_0_full_n(c_full_n[0]), .c_out_0_write(c_write[0]),
    .c_out_1_din(c_din[1]), .c_out_1_full_n(c_full_n[1]), .c_out_1_write(c_write[1]),
    .c_out_2_din(c_din[2]), .c_out_2_full_n(c_full_n[2]), .c_out_2_write(c_write[2]),
    .c_out_3_din(c_din[3]), .c_out_3_full_n(c_full_n[3]), .c_out_3_write(c_write[3]),
    .c_out_4_din(c_din[4]), .c_out_4_full_n(c_full_n[4]), .c_out_4_write(c_write[4]),
    .c_out_5_din(c_din[5]), .c_out_5_full_n(c_full_n[5]), .c_out_5_write(c_write[5]),
    .c_out_6_din(c_din[6]), .c_out_6_full_n(c_full_n[6]), .c_out_6_write(c_write[6]),
    .c_out_7_din(c_din[7]), .c_out_7_full_n(c_full_n[7]), .c_out_7_write(c_write[7]));

  initial begin
    if (!$value$plusargs("vecdir=%s", vecdir)) vecdir = "vectors";
    $readmemh({vecdir, "_a.hex"}, a_mem);
    $readmemh({vecdir, "_b.hex"}, b_mem);
    $readmemh({vecdir, "_c.hex"}, c_mem);
    for (i = 0; i < 8; i = i + 1) begin a_cnt[i] = 0; b_cnt[i] = 0; c_cnt[i] = 0; end
    for (p = 0; p < 16; p = p + 1) begin first_in[p] = -1; last_out[p] = -1; started[p] = 0; end
    cyc = 0; bad = 0; done_ports = 0;
    repeat (8) @(posedge ap_clk);
    ap_rst_n = 1;
  end

  always @(posedge ap_clk) if (ap_rst_n) begin
    cyc = cyc + 1;
    for (i = 0; i < 8; i = i + 1) begin
      if (a_read[i] && a_empty_n[i]) begin
        p = a_cnt[i] / STEPS;
        if (!started[p]) begin started[p] = 1; first_in[p] = cyc; end
        a_cnt[i] = a_cnt[i] + 1;
      end
      if (b_read[i] && b_empty_n[i]) begin
        p = b_cnt[i] / STEPS;
        if (!started[p]) begin started[p] = 1; first_in[p] = cyc; end
        b_cnt[i] = b_cnt[i] + 1;
      end
      if (c_write[i] && c_full_n[i]) begin
        p = c_cnt[i] / STEPS;
        if (c_din[i] !== c_mem[p * PER + i * STEPS + (c_cnt[i] % STEPS)]) begin
          if (bad < 8)
            $display("MISMATCH product %0d c_out_%0d[%0d]: got %0d want %0d",
                     p, i, c_cnt[i] % STEPS, $signed(c_din[i]),
                     $signed(c_mem[p * PER + i * STEPS + (c_cnt[i] % STEPS)]));
          bad = bad + 1;
        end
        last_out[p] = cyc;
        c_cnt[i] = c_cnt[i] + 1;
        if (c_cnt[i] == NPROD * STEPS) done_ports = done_ports + 1;
      end
    end
    if (done_ports == 8) begin
      $display("STUDY VALUES %0d checked, %0d wrong", NPROD * PER, bad);
      for (p = 0; p < NPROD; p = p + 1)
        $display("STUDY PRODUCT %0d first_in %0d last_out %0d latency %0d",
                 p, first_in[p], last_out[p], last_out[p] - first_in[p]);
      // Steady-state cost of one product: how far apart consecutive products
      // start once the pipeline is full. A design that finishes one product
      // before starting the next pays its whole latency here.
      for (p = 1; p < NPROD; p = p + 1)
        $display("STUDY INTERVAL %0d %0d", p, first_in[p] - first_in[p-1]);
      $display("STUDY RESULT %s", (bad == 0) ? "CORRECT" : "WRONG");
      $finish;
    end
    if (cyc > 20000) begin
      $display("STUDY VALUES %0d checked, %0d wrong", NPROD * PER, bad);
      $display("STUDY RESULT TIMEOUT after %0d cycles, %0d of %0d outputs seen",
               cyc, c_cnt[0] + c_cnt[1] + c_cnt[2] + c_cnt[3] + c_cnt[4] + c_cnt[5]
                    + c_cnt[6] + c_cnt[7], NPROD * PER);
      $finish;
    end
  end
endmodule
