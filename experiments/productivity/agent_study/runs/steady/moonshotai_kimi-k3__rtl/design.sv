// ---------------------------------------------------------------------------
// 8x8 output-stationary systolic matrix-multiply tile.
//
// Sixty-four processing elements PE(i,j) in an 8x8 grid. PE(i,j) accumulates
// C[i][j] = sum_k A[i][k]*B[k][j] in its own accumulator (output stationary;
// no partial sum ever leaves its element, no adder tree anywhere).
//
// A[i][k] enters the grid at PE(i,0) and travels EAST one element per cycle.
// B[k][j] enters the grid at PE(0,j) and travels SOUTH one element per cycle.
// A finished C[i][j] is captured into a drain register and travels WEST along
// its row, through the row's elements, to output port c_out_i. Every movement
// is between nearest neighbours only; nothing is broadcast.
//
// Schedule: a free-running 4-bit frame counter cyc (one product per 16-cycle
// frame). Port a_in_i is read while cyc in [i, i+7], so it carries A[i][0..7];
// port b_in_j is read while cyc in [j, j+7], so it carries B[0..7][j]. With
// this skew, A[i][k] and B[k][j] arrive at PE(i,j) on the same cycle. Each
// element registers the product (one multiplier -> one DSP) and accumulates
// it on the following cycle, so there is a single arithmetic operation
// between any two registers.
// ---------------------------------------------------------------------------

module pe (
  input  logic               ap_clk,
  input  logic               ap_rst_n,
  // A stream: in from the western neighbour, out to the eastern neighbour
  input  logic signed [7:0]  a_in,
  input  logic               a_in_v,
  input  logic               a_in_first, // this A value is A[i][0]
  input  logic               a_in_last,  // this A value is A[i][7]
  output logic signed [7:0]  a_out,
  output logic               a_out_v,
  output logic               a_out_first,
  output logic               a_out_last,
  // B stream: in from the northern neighbour, out to the southern neighbour
  input  logic signed [7:0]  b_in,
  input  logic               b_in_v,
  output logic signed [7:0]  b_out,
  output logic               b_out_v,
  // drain stream: finished results flow west along the row
  input  logic signed [31:0] d_in,   // from the eastern neighbour
  input  logic               d_in_v,
  output logic signed [31:0] d_out,  // to the western neighbour
  output logic               d_out_v
);

  logic signed [7:0]  a_reg, b_reg;
  logic               a_v, a_f, a_l, b_v;
  // one multiplier per element; forced into a DSP block
  (* use_dsp = "yes" *) logic signed [15:0] prod;
  logic               pv, pf, pl;
  logic signed [31:0] acc;
  logic               dload;
  logic signed [31:0] dreg;
  logic               dv;

  assign a_out       = a_reg;
  assign a_out_v     = a_v;
  assign a_out_first = a_f;
  assign a_out_last  = a_l;
  assign b_out       = b_reg;
  assign b_out_v     = b_v;
  assign d_out       = dreg;
  assign d_out_v     = dv;

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      a_reg <= '0; a_v <= 1'b0; a_f <= 1'b0; a_l <= 1'b0;
      b_reg <= '0; b_v <= 1'b0;
      prod  <= '0; pv <= 1'b0; pf <= 1'b0; pl <= 1'b0;
      acc   <= '0; dload <= 1'b0;
      dreg  <= '0; dv <= 1'b0;
    end else begin
      // nearest-neighbour pass: A one step east, B one step south
      a_reg <= a_in;
      a_v   <= a_in_v;
      a_f   <= a_in_first;
      a_l   <= a_in_last;
      b_reg <= b_in;
      b_v   <= b_in_v;
      // stage 1: this element's one multiplier (registered in/out -> DSP)
      prod <= a_reg * b_reg;
      pv   <= a_v & b_v;
      pf   <= a_f;
      pl   <= a_l;
      // stage 2: the accumulator; C[i][j] never leaves this element until done
      if (pv) acc <= pf ? prod : acc + prod;
      // one cycle after the last partial product lands, acc holds C[i][j]:
      // capture it into the drain register, which then shifts west
      dload <= pv & pl;
      if (dload) begin
        dreg <= acc;
        dv   <= 1'b1;
      end else begin
        dreg <= d_in;
        dv   <= d_in_v;
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

  // ---- frame counter: one product starts every 16 cycles -------------------
  logic [3:0] cyc;
  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) cyc <= 4'd0;
    else           cyc <= cyc + 4'd1;
  end

  // ---- per-port read strobes ------------------------------------------------
  logic [7:0] a_rd, b_rd;

  // ---- port data gathered into arrays ---------------------------------------
  wire [7:0] a_din [0:7];
  wire       a_en  [0:7];
  wire [7:0] b_din [0:7];
  wire       b_en  [0:7];

  assign a_din[0] = a_in_0_dout; assign a_en[0] = a_in_0_empty_n; assign a_in_0_read = a_rd[0];
  assign a_din[1] = a_in_1_dout; assign a_en[1] = a_in_1_empty_n; assign a_in_1_read = a_rd[1];
  assign a_din[2] = a_in_2_dout; assign a_en[2] = a_in_2_empty_n; assign a_in_2_read = a_rd[2];
  assign a_din[3] = a_in_3_dout; assign a_en[3] = a_in_3_empty_n; assign a_in_3_read = a_rd[3];
  assign a_din[4] = a_in_4_dout; assign a_en[4] = a_in_4_empty_n; assign a_in_4_read = a_rd[4];
  assign a_din[5] = a_in_5_dout; assign a_en[5] = a_in_5_empty_n; assign a_in_5_read = a_rd[5];
  assign a_din[6] = a_in_6_dout; assign a_en[6] = a_in_6_empty_n; assign a_in_6_read = a_rd[6];
  assign a_din[7] = a_in_7_dout; assign a_en[7] = a_in_7_empty_n; assign a_in_7_read = a_rd[7];

  assign b_din[0] = b_in_0_dout; assign b_en[0] = b_in_0_empty_n; assign b_in_0_read = b_rd[0];
  assign b_din[1] = b_in_1_dout; assign b_en[1] = b_in_1_empty_n; assign b_in_1_read = b_rd[1];
  assign b_din[2] = b_in_2_dout; assign b_en[2] = b_in_2_empty_n; assign b_in_2_read = b_rd[2];
  assign b_din[3] = b_in_3_dout; assign b_en[3] = b_in_3_empty_n; assign b_in_3_read = b_rd[3];
  assign b_din[4] = b_in_4_dout; assign b_en[4] = b_in_4_empty_n; assign b_in_4_read = b_rd[4];
  assign b_din[5] = b_in_5_dout; assign b_en[5] = b_in_5_empty_n; assign b_in_5_read = b_rd[5];
  assign b_din[6] = b_in_6_dout; assign b_en[6] = b_in_6_empty_n; assign b_in_6_read = b_rd[6];
  assign b_din[7] = b_in_7_dout; assign b_en[7] = b_in_7_empty_n; assign b_in_7_read = b_rd[7];

  // ---- inter-element wires ---------------------------------------------------
  // a_d[i][j] enters PE(i,j) from the west; a_d[i][j+1] leaves to the east.
  wire signed [7:0]  a_d [0:7][0:8];
  wire               a_v [0:7][0:8];
  wire               a_f [0:7][0:8];
  wire               a_l [0:7][0:8];
  // b_d[i][j] enters PE(i,j) from the north; b_d[i+1][j] leaves to the south.
  wire signed [7:0]  b_d [0:8][0:7];
  wire               b_v [0:8][0:7];
  // d_d[i][j] enters PE(i,j) from the east; d_d[i][j-1] leaves to the west.
  wire signed [31:0] d_d [0:7][0:8];
  wire               d_v [0:7][0:8];

  genvar i, j;
  generate
    for (i = 0; i < 8; i = i + 1) begin : row
      // read port i while cyc in [i, i+7]
      assign a_rd[i] = (cyc >= i) && (cyc <= i+7);
      assign b_rd[i] = (cyc >= i) && (cyc <= i+7);
      // A enters the grid at PE(i,0)
      assign a_d[i][0] = a_din[i];
      assign a_v[i][0] = a_rd[i] & a_en[i];
      assign a_f[i][0] = a_rd[i] & (cyc == i);
      assign a_l[i][0] = a_rd[i] & (cyc == i+7);
      // no result arrives from east of PE(i,7)
      assign d_d[i][8] = 32'sd0;
      assign d_v[i][8] = 1'b0;
      for (j = 0; j < 8; j = j + 1) begin : col
        pe u_pe (
          .ap_clk(ap_clk), .ap_rst_n(ap_rst_n),
          .a_in(a_d[i][j]),       .a_in_v(a_v[i][j]),
          .a_in_first(a_f[i][j]), .a_in_last(a_l[i][j]),
          .a_out(a_d[i][j+1]),    .a_out_v(a_v[i][j+1]),
          .a_out_first(a_f[i][j+1]), .a_out_last(a_l[i][j+1]),
          .b_in(b_d[i][j]),       .b_in_v(b_v[i][j]),
          .b_out(b_d[i+1][j]),    .b_out_v(b_v[i+1][j]),
          .d_in(d_d[i][j+1]),     .d_in_v(d_v[i][j+1]),
          .d_out(d_d[i][j]),      .d_out_v(d_v[i][j])
        );
      end
    end
  endgenerate

  // B enters the grid at PE(0,j)
  generate
    for (j = 0; j < 8; j = j + 1) begin : topedge
      assign b_d[0][j] = b_din[j];
      assign b_v[0][j] = b_rd[j] & b_en[j];
    end
  endgenerate

  // ---- outputs: each row's drain chain reaches its port through PE(i,0) -----
  assign c_out_0_din = d_d[0][0]; assign c_out_0_write = d_v[0][0];
  assign c_out_1_din = d_d[1][0]; assign c_out_1_write = d_v[1][0];
  assign c_out_2_din = d_d[2][0]; assign c_out_2_write = d_v[2][0];
  assign c_out_3_din = d_d[3][0]; assign c_out_3_write = d_v[3][0];
  assign c_out_4_din = d_d[4][0]; assign c_out_4_write = d_v[4][0];
  assign c_out_5_din = d_d[5][0]; assign c_out_5_write = d_v[5][0];
  assign c_out_6_din = d_d[6][0]; assign c_out_6_write = d_v[6][0];
  assign c_out_7_din = d_d[7][0]; assign c_out_7_write = d_v[7][0];

endmodule
