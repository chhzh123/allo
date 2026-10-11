`timescale 1ns/1ps
module tb_spiral;
  localparam N = 1024, W = 4, R = 256, NT = 8;
  reg clk = 0, reset = 0, next = 0;
  wire next_out;
  reg [31:0] stim [0:2*N*NT-1];
  reg [31:0] x [0:2*W-1];
  wire [31:0] y [0:2*W-1];
  dft_top dut(.clk(clk), .reset(reset), .next(next), .next_out(next_out), .X0(x[0]), .Y0(y[0]), .X1(x[1]), .Y1(y[1]), .X2(x[2]), .Y2(y[2]), .X3(x[3]), .Y3(y[3]), .X4(x[4]), .Y4(y[4]), .X5(x[5]), .Y5(y[5]), .X6(x[6]), .Y6(y[6]), .X7(x[7]), .Y7(y[7]));

  always #1.6665 clk = ~clk;
  integer cycle = 0;
  always @(posedge clk) cycle <= cycle + 1;

  integer first_in [0:NT-1];
  integer t, b, i, fo;
  initial begin
    $readmemh("stim.hex", stim);
    fo = $fopen("out.hex", "w");
    for (i = 0; i < 2*W; i = i + 1) x[i] = 0;
    @(posedge clk); reset <= 1;
    @(posedge clk); reset <= 0;
    @(posedge clk); @(posedge clk);
    next <= 1;                                   // the first vector starts on the following cycle
    @(posedge clk);
    for (t = 0; t < NT; t = t + 1) begin
      for (b = 0; b < R; b = b + 1) begin
        for (i = 0; i < 2*W; i = i + 1) x[i] <= stim[(t*N + b*W)*2 + i];
        next <= (b == R-1 && t < NT-1);          // announce the next vector with the last beat
        if (b == 0) first_in[t] = cycle + 1;     // edge at which the core samples this beat
        @(posedge clk);
      end
    end
    next <= 0;
    for (i = 0; i < 2*W; i = i + 1) x[i] <= 0;
  end

  reg cap = 0;
  integer ob = 0, ot = 0, fo_cycle = 0, j;
  always @(posedge clk) begin
    if (cap) begin
      for (j = 0; j < 2*W; j = j + 1) $fwrite(fo, "%08x\n", y[j]);
      if (ob == 0) fo_cycle = cycle;
      if (ob == R-1) begin
        $display("STAT %0d first_in %0d first_out %0d last_out %0d", ot, first_in[ot], fo_cycle, cycle);
        ot = ot + 1;
        ob = 0;
        cap <= next_out;
        if (ot == NT) begin $fclose(fo); $finish; end
      end else ob = ob + 1;
    end else if (next_out) cap <= 1;
  end

  initial begin #34809; $display("TIMEOUT"); $finish; end
endmodule
