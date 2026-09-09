// 8x8 output-stationary systolic matrix-multiply tile.
//
// Data movement: A travels east along row i, B travels south along column j.
// Input port i of A is skewed by i cycles and port j of B by j cycles, so
// A[i][k] and B[k][j] meet inside PE(i,j) at the same cycle and each PE
// accumulates its own C[i][j] over the eight values of k (one multiplier and
// one accumulator per PE, 64 in total).
//
// Each PE registers its product before accumulating it, so the multiplier is
// a clean registered 8x8 signed multiply that maps to one DSP block per PE.
// The registered product has exactly one consumer, the accumulator; the final
// result is a plain copy of the accumulator taken one cycle after the last
// addition, so no second multiply-add is ever inferred and the design uses
// exactly 64 DSP blocks.
//
// Draining: when a PE's accumulation finishes it starts a countdown of J+1
// cycles and then puts its result into the eastward result chain, which
// forwards one neighbour per cycle.  The countdown makes column j inject J+1
// cycles after column j-1, so the eight results of a row arrive at output
// port i in column order, one per cycle, without ever colliding.  The whole
// schedule is derived from the data wave itself, so it stays aligned whatever
// cycle the first transfer happens on.  One product is completed every 16
// cycles.

module pe #(
  parameter int I = 0,
  parameter int J = 0
) (
  input  logic               ap_clk,
  input  logic               ap_rst_n,
  // A operand arriving from the west (from the input port at J == 0)
  input  logic [7:0]         a_d,
  input  logic               a_v,
  // B operand arriving from the north (from the input port at I == 0)
  input  logic [7:0]         b_d,
  input  logic               b_v,
  // result arriving from the western neighbour (none at J == 0)
  input  logic signed [18:0] r_d,
  input  logic               r_v,
  // this PE's chain output, going east
  output logic signed [18:0] o_d,
  output logic               o_v,
  // A and B passed on to the eastern / southern neighbour
  output logic [7:0]         a_out,
  output logic               a_out_v,
  output logic [7:0]         b_out,
  output logic               b_out_v
);

  // 8 * 128 * 128 = 2^17 is the largest magnitude a sum can reach, so 19
  // signed bits are enough everywhere behind the output ports.
  logic signed [7:0]  a_reg, b_reg;
  logic               a_reg_v, b_reg_v;
  (* use_dsp = "yes" *) logic signed [15:0] prod_reg;
  logic               prod_v;
  (* use_dsp = "no" *)  logic signed [18:0] acc;
  logic [2:0]         kcnt;
  logic               mac_end, mac_end_d;
  logic signed [18:0] result;
  logic               pending;
  logic [3:0]         dcnt;
  logic               drain_now;

  assign mac_end   = prod_v && (kcnt == 3'd7);
  assign drain_now = (dcnt == 4'd1);

  assign a_out   = a_reg;
  assign a_out_v = a_reg_v;
  assign b_out   = b_reg;
  assign b_out_v = b_reg_v;

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      a_reg     <= '0;
      b_reg     <= '0;
      a_reg_v   <= 1'b0;
      b_reg_v   <= 1'b0;
      prod_reg  <= '0;
      prod_v    <= 1'b0;
      acc       <= '0;
      kcnt      <= '0;
      mac_end_d <= 1'b0;
      result    <= '0;
      pending   <= 1'b0;
      dcnt      <= '0;
      o_d       <= '0;
      o_v       <= 1'b0;
    end else begin
      // operand pipelines: one neighbour hop per cycle
      a_reg   <= a_d;
      a_reg_v <= a_v;
      b_reg   <= b_d;
      b_reg_v <= b_v;

      // the one multiplier of this PE, registered before it is added
      prod_reg <= a_reg * b_reg;
      prod_v   <= a_reg_v && b_reg_v;

      // the one accumulator of this PE, over the eight values of k
      if (prod_v) begin
        if (kcnt == 3'd0) acc <= prod_reg;
        else               acc <= acc + prod_reg;
        kcnt <= kcnt + 3'd1;
      end else begin
        kcnt <= 3'd0;
      end

      // one cycle after the last addition the sum is complete: copy it out
      mac_end_d <= mac_end;
      if (mac_end_d) begin
        result  <= acc;
        pending <= 1'b1;
      end

      // schedule this PE's turn in the eastward drain chain
      if (mac_end_d)       dcnt <= 4'(J + 1);
      else if (dcnt != 0)  dcnt <= dcnt - 4'd1;

      // inject own result when its turn comes, otherwise pass through
      if (drain_now && pending) begin
        o_d     <= result;
        o_v     <= 1'b1;
        pending <= 1'b0;
      end else if (r_v) begin
        o_d <= r_d;
        o_v <= 1'b1;
      end else begin
        o_v <= 1'b0;
      end
    end
  end

endmodule


module dut_norm (
  input wire ap_clk, input wire ap_rst_n,
  input wire [7:0] a_in_0_dout, input wire a_in_0_empty_n, output wire a_in_0_read,
  input wire [7:0] a_in_1_dout, input wire a_in_1_empty_n, output wire a_in_1_read,
  input wire [7:0] a_in_2_dout, input wire a_in_2_empty_n, output wire a_in_2_read,
  input wire [7:0] a_in_3_dout, input wire a_in_3_empty_n, output wire a_in_3_read,
  input wire [7:0] a_in_4_dout, input wire a_in_4_empty_n, output wire a_in_4_read,
  input wire [7:0] a_in_5_dout, input wire a_in_5_empty_n, output wire a_in_5_read,
  input wire [7:0] a_in_6_dout, input wire a_in_6_empty_n, output wire a_in_6_read,
  input wire [7:0] a_in_7_dout, input wire a_in_7_empty_n, output wire a_in_7_read,
  input wire [7:0] b_in_0_dout, input wire b_in_0_empty_n, output wire b_in_0_read,
  input wire [7:0] b_in_1_dout, input wire b_in_1_empty_n, output wire b_in_1_read,
  input wire [7:0] b_in_2_dout, input wire b_in_2_empty_n, output wire b_in_2_read,
  input wire [7:0] b_in_3_dout, input wire b_in_3_empty_n, output wire b_in_3_read,
  input wire [7:0] b_in_4_dout, input wire b_in_4_empty_n, output wire b_in_4_read,
  input wire [7:0] b_in_5_dout, input wire b_in_5_empty_n, output wire b_in_5_read,
  input wire [7:0] b_in_6_dout, input wire b_in_6_empty_n, output wire b_in_6_read,
  input wire [7:0] b_in_7_dout, input wire b_in_7_empty_n, output wire b_in_7_read,
  output wire [31:0] c_out_0_din, input wire c_out_0_full_n, output wire c_out_0_write,
  output wire [31:0] c_out_1_din, input wire c_out_1_full_n, output wire c_out_1_write,
  output wire [31:0] c_out_2_din, input wire c_out_2_full_n, output wire c_out_2_write,
  output wire [31:0] c_out_3_din, input wire c_out_3_full_n, output wire c_out_3_write,
  output wire [31:0] c_out_4_din, input wire c_out_4_full_n, output wire c_out_4_write,
  output wire [31:0] c_out_5_din, input wire c_out_5_full_n, output wire c_out_5_write,
  output wire [31:0] c_out_6_din, input wire c_out_6_full_n, output wire c_out_6_write,
  output wire [31:0] c_out_7_din, input wire c_out_7_full_n, output wire c_out_7_write
);

  // ------------------------------------------------------------------
  // ports
  // ------------------------------------------------------------------
  wire [7:0] a_port_d [0:7];
  wire       a_port_e [0:7];
  wire       a_port_r [0:7];
  wire [7:0] b_port_d [0:7];
  wire       b_port_e [0:7];
  wire       b_port_r [0:7];
  wire signed [18:0] c_port_d [0:7];
  wire              c_port_w [0:7];

  assign a_port_d[0] = a_in_0_dout; assign a_port_e[0] = a_in_0_empty_n; assign a_in_0_read = a_port_r[0];
  assign a_port_d[1] = a_in_1_dout; assign a_port_e[1] = a_in_1_empty_n; assign a_in_1_read = a_port_r[1];
  assign a_port_d[2] = a_in_2_dout; assign a_port_e[2] = a_in_2_empty_n; assign a_in_2_read = a_port_r[2];
  assign a_port_d[3] = a_in_3_dout; assign a_port_e[3] = a_in_3_empty_n; assign a_in_3_read = a_port_r[3];
  assign a_port_d[4] = a_in_4_dout; assign a_port_e[4] = a_in_4_empty_n; assign a_in_4_read = a_port_r[4];
  assign a_port_d[5] = a_in_5_dout; assign a_port_e[5] = a_in_5_empty_n; assign a_in_5_read = a_port_r[5];
  assign a_port_d[6] = a_in_6_dout; assign a_port_e[6] = a_in_6_empty_n; assign a_in_6_read = a_port_r[6];
  assign a_port_d[7] = a_in_7_dout; assign a_port_e[7] = a_in_7_empty_n; assign a_in_7_read = a_port_r[7];

  assign b_port_d[0] = b_in_0_dout; assign b_port_e[0] = b_in_0_empty_n; assign b_in_0_read = b_port_r[0];
  assign b_port_d[1] = b_in_1_dout; assign b_port_e[1] = b_in_1_empty_n; assign b_in_1_read = b_port_r[1];
  assign b_port_d[2] = b_in_2_dout; assign b_port_e[2] = b_in_2_empty_n; assign b_in_2_read = b_port_r[2];
  assign b_port_d[3] = b_in_3_dout; assign b_port_e[3] = b_in_3_empty_n; assign b_in_3_read = b_port_r[3];
  assign b_port_d[4] = b_in_4_dout; assign b_port_e[4] = b_in_4_empty_n; assign b_in_4_read = b_port_r[4];
  assign b_port_d[5] = b_in_5_dout; assign b_port_e[5] = b_in_5_empty_n; assign b_in_5_read = b_port_r[5];
  assign b_port_d[6] = b_in_6_dout; assign b_port_e[6] = b_in_6_empty_n; assign b_in_6_read = b_port_r[6];
  assign b_port_d[7] = b_in_7_dout; assign b_port_e[7] = b_in_7_empty_n; assign b_in_7_read = b_port_r[7];

  assign c_out_0_din = {{13{c_port_d[0][18]}}, c_port_d[0]}; assign c_out_0_write = c_port_w[0];
  assign c_out_1_din = {{13{c_port_d[1][18]}}, c_port_d[1]}; assign c_out_1_write = c_port_w[1];
  assign c_out_2_din = {{13{c_port_d[2][18]}}, c_port_d[2]}; assign c_out_2_write = c_port_w[2];
  assign c_out_3_din = {{13{c_port_d[3][18]}}, c_port_d[3]}; assign c_out_3_write = c_port_w[3];
  assign c_out_4_din = {{13{c_port_d[4][18]}}, c_port_d[4]}; assign c_out_4_write = c_port_w[4];
  assign c_out_5_din = {{13{c_port_d[5][18]}}, c_port_d[5]}; assign c_out_5_write = c_port_w[5];
  assign c_out_6_din = {{13{c_port_d[6][18]}}, c_port_d[6]}; assign c_out_6_write = c_port_w[6];
  assign c_out_7_din = {{13{c_port_d[7][18]}}, c_port_d[7]}; assign c_out_7_write = c_port_w[7];

  // ------------------------------------------------------------------
  // input schedule: a 16-cycle product period, port i of A skewed by i
  // cycles and port j of B by j cycles, so that A[i][k] and B[k][j] meet
  // inside PE(i,j).  Port s is read during phases [s, s+7] (mod 16).
  // ------------------------------------------------------------------
  logic [3:0] cnt;
  always_ff @(posedge ap_clk)
    if (!ap_rst_n) cnt <= 4'd0;
    else           cnt <= cnt + 4'd1;

  wire [3:0] askew [0:7];
  wire [3:0] bskew [0:7];
  genvar s;
  generate
    for (s = 0; s < 8; s = s + 1) begin : skew
      assign askew[s] = cnt - 4'(s);
      assign bskew[s] = cnt - 4'(s);
      assign a_port_r[s] = ~askew[s][3];
      assign b_port_r[s] = ~bskew[s][3];
    end
  endgenerate

  // ------------------------------------------------------------------
  // inter-PE wires: each PE talks only to its nearest neighbours
  // ------------------------------------------------------------------
  wire [7:0]  a_dat [0:7][0:8];   // A entering PE(i,j) from the west
  wire        a_val [0:7][0:8];
  wire [7:0]  b_dat [0:8][0:7];   // B entering PE(i,j) from the north
  wire        b_val [0:8][0:7];
  wire signed [18:0] r_dat [0:7][0:8];  // result entering PE(i,j) from the west
  wire               r_val [0:7][0:8];
  wire signed [18:0] o_dat [0:7][0:8];  // result leaving PE(i,j) eastward
  wire               o_val [0:7][0:8];

  genvar i, j;
  generate
    for (i = 0; i < 8; i = i + 1) begin : row
      // A enters row i at column 0 straight from its input port
      assign a_dat[i][0] = a_port_d[i];
      assign a_val[i][0] = a_port_r[i] & a_port_e[i];
      assign r_dat[i][0] = 19'sd0;
      assign r_val[i][0] = 1'b0;
      for (j = 0; j < 8; j = j + 1) begin : col
        pe #(.I(i), .J(j)) u_pe (
          .ap_clk   (ap_clk),
          .ap_rst_n (ap_rst_n),
          .a_d      (a_dat[i][j]),
          .a_v      (a_val[i][j]),
          .b_d      (b_dat[i][j]),
          .b_v      (b_val[i][j]),
          .r_d      (r_dat[i][j]),
          .r_v      (r_val[i][j]),
          .o_d      (o_dat[i][j]),
          .o_v      (o_val[i][j]),
          .a_out    (a_dat[i][j+1]),
          .a_out_v  (a_val[i][j+1]),
          .b_out    (b_dat[i+1][j]),
          .b_out_v  (b_val[i+1][j])
        );
        // results flow east along the row, one neighbour per cycle
        assign r_dat[i][j+1] = o_dat[i][j];
        assign r_val[i][j+1] = o_val[i][j];
      end
      // column 7 of row i drives output port i
      assign c_port_d[i] = o_dat[i][7];
      assign c_port_w[i] = o_val[i][7];
      // B enters column i at row 0 straight from its input port
      assign b_dat[0][i] = b_port_d[i];
      assign b_val[0][i] = b_port_r[i] & b_port_e[i];
    end
  endgenerate

endmodule
