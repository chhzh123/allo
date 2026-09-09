// Processing element: one multiplier, one accumulator, output-stationary.
// Pipeline: register A/B -> register product -> accumulate.
// Registered product ensures DSP48 inference.
module pe (
  input ap_clk, ap_rst_n,
  input  wire [7:0] a_in,  input  wire a_in_valid,
  output wire [7:0] a_out, output wire a_out_valid,
  input  wire [7:0] b_in,  input  wire b_in_valid,
  output wire [7:0] b_out, output wire b_out_valid,
  input  wire [31:0] r_in,
  output wire [31:0] r_out,
  input  wire load,
  input  wire shift_en
);

  // Stage 1: registered inputs for pass-through
  logic signed [7:0] a_reg, b_reg;
  logic a_valid_reg, b_valid_reg;

  // Stage 2: registered product
  logic signed [15:0] prod;
  logic prod_valid;

  // Stage 3: accumulator
  logic signed [31:0] acc;
  logic [2:0] count;

  // Result hold register
  logic signed [31:0] result_reg;

  // Shift register for drain
  logic signed [31:0] shift_reg;

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      a_reg        <= 0;
      a_valid_reg  <= 0;
      b_reg        <= 0;
      b_valid_reg  <= 0;
      prod         <= 0;
      prod_valid   <= 0;
      acc          <= 0;
      count        <= 0;
      result_reg   <= 0;
      shift_reg    <= 0;
    end else begin
      // ---- Stage 1: register inputs ----
      a_reg       <= a_in;
      a_valid_reg <= a_in_valid;
      b_reg       <= b_in;
      b_valid_reg <= b_in_valid;

      // ---- Stage 2: register product ----
      prod       <= $signed(a_reg) * $signed(b_reg);
      prod_valid <= a_valid_reg && b_valid_reg;

      // ---- Stage 3: accumulate ----
      if (prod_valid) begin
        if (count == 3'd7) begin
          result_reg <= acc + prod;
          acc        <= 0;
          count      <= 0;
        end else begin
          acc   <= acc + prod;
          count <= count + 1;
        end
      end

      // ---- Drain shift register ----
      if (load) begin
        shift_reg <= result_reg;
      end else if (shift_en) begin
        shift_reg <= r_in;
      end
    end
  end

  assign a_out       = a_reg;
  assign a_out_valid = a_valid_reg;
  assign b_out       = b_reg;
  assign b_out_valid = b_valid_reg;
  assign r_out       = shift_reg;

endmodule


// Top-level 8x8 output-stationary systolic matrix-multiply tile
module dut_norm (
  input wire ap_clk, input wire ap_rst_n,
  input  wire [7:0] a_in_0_dout, input wire a_in_0_empty_n, output wire a_in_0_read,
  input  wire [7:0] a_in_1_dout, input wire a_in_1_empty_n, output wire a_in_1_read,
  input  wire [7:0] a_in_2_dout, input wire a_in_2_empty_n, output wire a_in_2_read,
  input  wire [7:0] a_in_3_dout, input wire a_in_3_empty_n, output wire a_in_3_read,
  input  wire [7:0] a_in_4_dout, input wire a_in_4_empty_n, output wire a_in_4_read,
  input  wire [7:0] a_in_5_dout, input wire a_in_5_empty_n, output wire a_in_5_read,
  input  wire [7:0] a_in_6_dout, input wire a_in_6_empty_n, output wire a_in_6_read,
  input  wire [7:0] a_in_7_dout, input wire a_in_7_empty_n, output wire a_in_7_read,
  input  wire [7:0] b_in_0_dout, input wire b_in_0_empty_n, output wire b_in_0_read,
  input  wire [7:0] b_in_1_dout, input wire b_in_1_empty_n, output wire b_in_1_read,
  input  wire [7:0] b_in_2_dout, input wire b_in_2_empty_n, output wire b_in_2_read,
  input  wire [7:0] b_in_3_dout, input wire b_in_3_empty_n, output wire b_in_3_read,
  input  wire [7:0] b_in_4_dout, input wire b_in_4_empty_n, output wire b_in_4_read,
  input  wire [7:0] b_in_5_dout, input wire b_in_5_empty_n, output wire b_in_5_read,
  input  wire [7:0] b_in_6_dout, input wire b_in_6_empty_n, output wire b_in_6_read,
  input  wire [7:0] b_in_7_dout, input wire b_in_7_empty_n, output wire b_in_7_read,
  output wire [31:0] c_out_0_din, input wire c_out_0_full_n, output wire c_out_0_write,
  output wire [31:0] c_out_1_din, input wire c_out_1_full_n, output wire c_out_1_write,
  output wire [31:0] c_out_2_din, input wire c_out_2_full_n, output wire c_out_2_write,
  output wire [31:0] c_out_3_din, input wire c_out_3_full_n, output wire c_out_3_write,
  output wire [31:0] c_out_4_din, input wire c_out_4_full_n, output wire c_out_4_write,
  output wire [31:0] c_out_5_din, input wire c_out_5_full_n, output wire c_out_5_write,
  output wire [31:0] c_out_6_din, input wire c_out_6_full_n, output wire c_out_6_write,
  output wire [31:0] c_out_7_din, input wire c_out_7_full_n, output wire c_out_7_write
);

  // ---- Global cycle counter ----
  logic [15:0] cycle;
  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) cycle <= 0;
    else cycle <= cycle + 1;
  end

  // ---- Interconnect wires (one larger in direction of travel) ----
  wire [7:0]  a_h       [0:7][0:8];  // A travels east
  wire        a_h_valid [0:7][0:8];
  wire [7:0]  b_v       [0:8][0:7];  // B travels south
  wire        b_v_valid [0:8][0:7];
  wire [31:0] r_w       [0:7][0:8];  // results travel west; [*][8] tied to 0

  // ---- Per-row control signals ----
  // With registered product, PE(i,j) result_reg ready at cycle i+j+10.
  // PE(i,7) ready at i+17. Load at i+18 to avoid same-cycle conflict.
  wire load    [0:7];
  wire shift_en [0:7];

  genvar i, j;
  generate
    for (i = 0; i < 8; i++) begin : row_ctrl
      assign load[i]    = (cycle >= (i + 18)) && (cycle[2:0] == i[2:0]);
      assign shift_en[i] = (cycle >= (i + 18)) && (cycle[2:0] != i[2:0]);
    end
  endgenerate

  // ---- Input connections ----
  assign a_h[0][0] = a_in_0_dout;
  assign a_h[1][0] = a_in_1_dout;
  assign a_h[2][0] = a_in_2_dout;
  assign a_h[3][0] = a_in_3_dout;
  assign a_h[4][0] = a_in_4_dout;
  assign a_h[5][0] = a_in_5_dout;
  assign a_h[6][0] = a_in_6_dout;
  assign a_h[7][0] = a_in_7_dout;

  assign a_h_valid[0][0] = (cycle >= 0);
  assign a_h_valid[1][0] = (cycle >= 1);
  assign a_h_valid[2][0] = (cycle >= 2);
  assign a_h_valid[3][0] = (cycle >= 3);
  assign a_h_valid[4][0] = (cycle >= 4);
  assign a_h_valid[5][0] = (cycle >= 5);
  assign a_h_valid[6][0] = (cycle >= 6);
  assign a_h_valid[7][0] = (cycle >= 7);

  assign b_v[0][0] = b_in_0_dout;
  assign b_v[0][1] = b_in_1_dout;
  assign b_v[0][2] = b_in_2_dout;
  assign b_v[0][3] = b_in_3_dout;
  assign b_v[0][4] = b_in_4_dout;
  assign b_v[0][5] = b_in_5_dout;
  assign b_v[0][6] = b_in_6_dout;
  assign b_v[0][7] = b_in_7_dout;

  assign b_v_valid[0][0] = (cycle >= 0);
  assign b_v_valid[0][1] = (cycle >= 1);
  assign b_v_valid[0][2] = (cycle >= 2);
  assign b_v_valid[0][3] = (cycle >= 3);
  assign b_v_valid[0][4] = (cycle >= 4);
  assign b_v_valid[0][5] = (cycle >= 5);
  assign b_v_valid[0][6] = (cycle >= 6);
  assign b_v_valid[0][7] = (cycle >= 7);

  // ---- Input read signals (driven from registered state) ----
  assign a_in_0_read = (cycle >= 0);
  assign a_in_1_read = (cycle >= 1);
  assign a_in_2_read = (cycle >= 2);
  assign a_in_3_read = (cycle >= 3);
  assign a_in_4_read = (cycle >= 4);
  assign a_in_5_read = (cycle >= 5);
  assign a_in_6_read = (cycle >= 6);
  assign a_in_7_read = (cycle >= 7);

  assign b_in_0_read = (cycle >= 0);
  assign b_in_1_read = (cycle >= 1);
  assign b_in_2_read = (cycle >= 2);
  assign b_in_3_read = (cycle >= 3);
  assign b_in_4_read = (cycle >= 4);
  assign b_in_5_read = (cycle >= 5);
  assign b_in_6_read = (cycle >= 6);
  assign b_in_7_read = (cycle >= 7);

  // ---- Tie off eastern result inputs ----
  generate
    for (i = 0; i < 8; i++) begin : tie_r
      assign r_w[i][8] = 0;
    end
  endgenerate

  // ---- 8x8 PE grid ----
  generate
    for (i = 0; i < 8; i++) begin : row
      for (j = 0; j < 8; j++) begin : col
        pe u_pe (
          .ap_clk      (ap_clk),
          .ap_rst_n    (ap_rst_n),
          .a_in        (a_h[i][j]),
          .a_in_valid  (a_h_valid[i][j]),
          .a_out       (a_h[i][j+1]),
          .a_out_valid (a_h_valid[i][j+1]),
          .b_in        (b_v[i][j]),
          .b_in_valid  (b_v_valid[i][j]),
          .b_out       (b_v[i+1][j]),
          .b_out_valid (b_v_valid[i+1][j]),
          .r_in        (r_w[i][j+1]),
          .r_out       (r_w[i][j]),
          .load        (load[i]),
          .shift_en    (shift_en[i])
        );
      end
    end
  endgenerate

  // ---- Output connections ----
  assign c_out_0_din = r_w[0][0];
  assign c_out_1_din = r_w[1][0];
  assign c_out_2_din = r_w[2][0];
  assign c_out_3_din = r_w[3][0];
  assign c_out_4_din = r_w[4][0];
  assign c_out_5_din = r_w[5][0];
  assign c_out_6_din = r_w[6][0];
  assign c_out_7_din = r_w[7][0];

  // ---- Output write signals (driven from registered state) ----
  logic c_out_write_reg [0:7];
  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      c_out_write_reg[0] <= 0;
      c_out_write_reg[1] <= 0;
      c_out_write_reg[2] <= 0;
      c_out_write_reg[3] <= 0;
      c_out_write_reg[4] <= 0;
      c_out_write_reg[5] <= 0;
      c_out_write_reg[6] <= 0;
      c_out_write_reg[7] <= 0;
    end else begin
      c_out_write_reg[0] <= (cycle >= 18);
      c_out_write_reg[1] <= (cycle >= 19);
      c_out_write_reg[2] <= (cycle >= 20);
      c_out_write_reg[3] <= (cycle >= 21);
      c_out_write_reg[4] <= (cycle >= 22);
      c_out_write_reg[5] <= (cycle >= 23);
      c_out_write_reg[6] <= (cycle >= 24);
      c_out_write_reg[7] <= (cycle >= 25);
    end
  end

  assign c_out_0_write = c_out_write_reg[0];
  assign c_out_1_write = c_out_write_reg[1];
  assign c_out_2_write = c_out_write_reg[2];
  assign c_out_3_write = c_out_write_reg[3];
  assign c_out_4_write = c_out_write_reg[4];
  assign c_out_5_write = c_out_write_reg[5];
  assign c_out_6_write = c_out_write_reg[6];
  assign c_out_7_write = c_out_write_reg[7];

endmodule