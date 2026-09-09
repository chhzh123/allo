// ---------------------------------------------------------------------------
// 8x8 output-stationary systolic matrix-multiply tile.
//
//   * 64 processing elements, one multiplier + one accumulator each.
//   * A flows east   (PE(i,0) -> PE(i,1) -> ... -> PE(i,7))
//   * B flows south  (PE(0,j) -> PE(1,j) -> ... -> PE(7,j))
//   * C[i][j] is accumulated in place in PE(i,j) and drained east along the
//     row through a per-row chain of registers, leaving via PE(i,7).
//   * Every hop is a register: one neighbour per cycle, nothing broadcast.
//     Even the accumulate/drain control travels with the A operand as two
//     extra flag bits (first-k, last-k), so no element sees a global signal
//     other than clock and reset.
//
// Skew is produced at the ports: a_in_i begins transferring at cycle i+1 and
// b_in_j at cycle j+1, so A[i][k] and B[k][j] arrive at PE(i,j) in the same
// cycle.  Each port then reads one value every cycle for as long as data is
// offered, so products are issued back to back every 8 cycles.
//
// Cycle accounting (cycle 0 = first cycle with reset released):
//   A[i][k] presented by port i   during cycle i+k+2
//   a_r/b_r aligned at PE(i,j)    during cycle i+j+k+3
//   product registered            during cycle i+j+k+4
//   accumulator holds sum..k      during cycle i+j+k+5
//   result captured into `hold`   at    cycle i+j+12   (k=7)
//   pushed onto the row chain     at    cycle i+2j+13
//   emerges at output port i      during cycle i+j+21
// so row i emits C[i][0..7] on eight consecutive cycles, in order, and the
// last value of the first product leaves during cycle 35.  Every flag is
// qualified by its valid bit, so the array goes quiet when input stops and
// never emits a value it was not given data for.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Input port stage: waits DLY cycles out of reset, then transfers one value
// per cycle and tags it with its k-index flags.
// ---------------------------------------------------------------------------
module mm_in_port #(parameter int DLY = 0) (
  input  wire        ap_clk,
  input  wire        ap_rst_n,
  input  wire [7:0]  dout,
  input  wire        empty_n,
  output wire        read,
  output logic [7:0] q,
  output logic       qf,
  output logic       ql,
  output logic       qv
);
  logic [3:0] dly;
  logic       started;
  logic [2:0] k;

  // driven from registered state only, never from empty_n
  assign read = started;

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      dly     <= DLY[3:0];
      started <= 1'b0;
      k       <= 3'd0;
      q       <= 8'd0;
      qf      <= 1'b0;
      ql      <= 1'b0;
      qv      <= 1'b0;
    end else begin
      if (!started) begin
        if (dly == 4'd0) started <= 1'b1;
        else             dly     <= dly - 4'd1;
      end
      // no transfer this cycle means no valid token and no live flags
      qv <= 1'b0;
      qf <= 1'b0;
      ql <= 1'b0;
      if (started && empty_n) begin
        q  <= dout;
        qf <= (k == 3'd0);
        ql <= (k == 3'd7);
        qv <= 1'b1;
        k  <= k + 3'd1;
      end
    end
  end
endmodule

// ---------------------------------------------------------------------------
// Processing element.  J is the column index; it only sets how long this
// element waits before pushing its own result onto its row's drain chain.
// ---------------------------------------------------------------------------
module mm_pe #(parameter int J = 0) (
  input  wire        ap_clk,
  input  wire        ap_rst_n,
  // A operand: from the western neighbour, to the eastern neighbour
  input  wire [7:0]  a_in,
  input  wire        a_f_in,
  input  wire        a_l_in,
  input  wire        a_v_in,
  output wire [7:0]  a_out,
  output wire        a_f_out,
  output wire        a_l_out,
  output wire        a_v_out,
  // B operand: from the northern neighbour, to the southern neighbour
  input  wire [7:0]  b_in,
  output wire [7:0]  b_out,
  // result drain chain: from the western neighbour, to the eastern neighbour
  input  wire [31:0] c_in,
  input  wire        c_v_in,
  output wire [31:0] c_out,
  output wire        c_v_out
);
  // operand registers (one neighbour hop per cycle)
  logic signed [7:0]  a_r, b_r;
  logic               a_f_r, a_l_r, a_v_r;

  // multiplier output register: one arithmetic operation between registers
  (* use_dsp = "yes" *) logic signed [15:0] m;
  logic                                    m_f, m_l, m_v;

  // the stationary accumulator
  logic signed [31:0] acc;

  // finished result, held while the next product accumulates behind it
  logic signed [31:0] hold;
  logic               l2;

  // local shift register: this column injects J cycles after `hold` is ready
  logic [7:0]         pipe;

  // this element's slot on its row's drain chain
  logic signed [31:0] c_r;
  logic               c_v_r;

  wire signed [31:0] mx = {{16{m[15]}}, m};

  assign a_out   = a_r;
  assign a_f_out = a_f_r;
  assign a_l_out = a_l_r;
  assign a_v_out = a_v_r;
  assign b_out   = b_r;
  assign c_out   = c_r;
  assign c_v_out = c_v_r;

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      a_r   <= 8'sd0;  b_r   <= 8'sd0;
      a_f_r <= 1'b0;   a_l_r <= 1'b0;  a_v_r <= 1'b0;
      m     <= 16'sd0;
      m_f   <= 1'b0;   m_l   <= 1'b0;  m_v <= 1'b0;
      acc   <= 32'sd0;
      hold  <= 32'sd0;
      l2    <= 1'b0;
      pipe  <= 8'd0;
      c_r   <= 32'sd0;
      c_v_r <= 1'b0;
    end else begin
      // ---- nearest-neighbour operand movement --------------------------
      a_r   <= a_in;
      a_f_r <= a_f_in && a_v_in;
      a_l_r <= a_l_in && a_v_in;
      a_v_r <= a_v_in;
      b_r   <= b_in;

      // ---- multiply, registered before it is ever added ----------------
      if (a_v_r) m <= a_r * b_r;
      m_f <= a_f_r && a_v_r;
      m_l <= a_l_r && a_v_r;
      m_v <= a_v_r;

      // ---- accumulate in place; k==0 restarts the sum ------------------
      if (m_v) acc <= m_f ? mx : (acc + mx);

      // ---- capture the finished element one cycle after the last add ---
      l2 <= m_l;
      if (l2) hold <= acc;

      // ---- local delay so column J injects into its own free slot ------
      pipe <= {pipe[6:0], l2};

      // ---- row drain chain, moving east --------------------------------
      if (pipe[J]) begin
        c_r   <= hold;
        c_v_r <= 1'b1;
      end else begin
        c_r   <= c_in;
        c_v_r <= c_v_in;
      end
    end
  end
endmodule

// ---------------------------------------------------------------------------
// Top level
// ---------------------------------------------------------------------------
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

  // ---------------- flat ports gathered into arrays -----------------------
  wire [7:0]  a_dv [0:7];
  wire        a_ev [0:7];
  wire        a_rv [0:7];
  wire [7:0]  b_dv [0:7];
  wire        b_ev [0:7];
  wire        b_rv [0:7];
  wire [31:0] c_dv [0:7];
  wire        c_wv [0:7];

  assign a_dv[0] = a_in_0_dout; assign a_ev[0] = a_in_0_empty_n; assign a_in_0_read = a_rv[0];
  assign a_dv[1] = a_in_1_dout; assign a_ev[1] = a_in_1_empty_n; assign a_in_1_read = a_rv[1];
  assign a_dv[2] = a_in_2_dout; assign a_ev[2] = a_in_2_empty_n; assign a_in_2_read = a_rv[2];
  assign a_dv[3] = a_in_3_dout; assign a_ev[3] = a_in_3_empty_n; assign a_in_3_read = a_rv[3];
  assign a_dv[4] = a_in_4_dout; assign a_ev[4] = a_in_4_empty_n; assign a_in_4_read = a_rv[4];
  assign a_dv[5] = a_in_5_dout; assign a_ev[5] = a_in_5_empty_n; assign a_in_5_read = a_rv[5];
  assign a_dv[6] = a_in_6_dout; assign a_ev[6] = a_in_6_empty_n; assign a_in_6_read = a_rv[6];
  assign a_dv[7] = a_in_7_dout; assign a_ev[7] = a_in_7_empty_n; assign a_in_7_read = a_rv[7];

  assign b_dv[0] = b_in_0_dout; assign b_ev[0] = b_in_0_empty_n; assign b_in_0_read = b_rv[0];
  assign b_dv[1] = b_in_1_dout; assign b_ev[1] = b_in_1_empty_n; assign b_in_1_read = b_rv[1];
  assign b_dv[2] = b_in_2_dout; assign b_ev[2] = b_in_2_empty_n; assign b_in_2_read = b_rv[2];
  assign b_dv[3] = b_in_3_dout; assign b_ev[3] = b_in_3_empty_n; assign b_in_3_read = b_rv[3];
  assign b_dv[4] = b_in_4_dout; assign b_ev[4] = b_in_4_empty_n; assign b_in_4_read = b_rv[4];
  assign b_dv[5] = b_in_5_dout; assign b_ev[5] = b_in_5_empty_n; assign b_in_5_read = b_rv[5];
  assign b_dv[6] = b_in_6_dout; assign b_ev[6] = b_in_6_empty_n; assign b_in_6_read = b_rv[6];
  assign b_dv[7] = b_in_7_dout; assign b_ev[7] = b_in_7_empty_n; assign b_in_7_read = b_rv[7];

  assign c_out_0_din = c_dv[0]; assign c_out_0_write = c_wv[0];
  assign c_out_1_din = c_dv[1]; assign c_out_1_write = c_wv[1];
  assign c_out_2_din = c_dv[2]; assign c_out_2_write = c_wv[2];
  assign c_out_3_din = c_dv[3]; assign c_out_3_write = c_wv[3];
  assign c_out_4_din = c_dv[4]; assign c_out_4_write = c_wv[4];
  assign c_out_5_din = c_dv[5]; assign c_out_5_write = c_wv[5];
  assign c_out_6_din = c_dv[6]; assign c_out_6_write = c_wv[6];
  assign c_out_7_din = c_dv[7]; assign c_out_7_write = c_wv[7];

  // ---------------- port stages (these produce the input skew) ------------
  wire [7:0] aq  [0:7];
  wire       aqf [0:7];
  wire       aql [0:7];
  wire       aqv [0:7];
  wire [7:0] bq  [0:7];

  genvar i, j;
  generate
    for (i = 0; i < 8; i = i + 1) begin : g_a_port
      mm_in_port #(.DLY(i)) u_ap (
        .ap_clk(ap_clk), .ap_rst_n(ap_rst_n),
        .dout(a_dv[i]), .empty_n(a_ev[i]), .read(a_rv[i]),
        .q(aq[i]), .qf(aqf[i]), .ql(aql[i]), .qv(aqv[i]));
    end
    for (j = 0; j < 8; j = j + 1) begin : g_b_port
      wire bf_unused, bl_unused, bv_unused;
      mm_in_port #(.DLY(j)) u_bp (
        .ap_clk(ap_clk), .ap_rst_n(ap_rst_n),
        .dout(b_dv[j]), .empty_n(b_ev[j]), .read(b_rv[j]),
        .q(bq[j]), .qf(bf_unused), .ql(bl_unused), .qv(bv_unused));
    end
  endgenerate

  // ---------------- nearest-neighbour interconnect ------------------------
  wire [7:0]  AH [0:7][0:8];   // A, west -> east
  wire        AF [0:7][0:8];
  wire        AL [0:7][0:8];
  wire        AV [0:7][0:8];
  wire [7:0]  BV [0:8][0:7];   // B, north -> south
  wire [31:0] CH [0:7][0:8];   // C, west -> east
  wire        CW [0:7][0:8];

  generate
    for (i = 0; i < 8; i = i + 1) begin : g_row_edge
      assign AH[i][0] = aq[i];
      assign AF[i][0] = aqf[i];
      assign AL[i][0] = aql[i];
      assign AV[i][0] = aqv[i];
      assign CH[i][0] = 32'd0;      // nothing enters the drain chain here
      assign CW[i][0] = 1'b0;
      assign c_dv[i]  = CH[i][8];   // output port i is driven by PE(i,7)
      assign c_wv[i]  = CW[i][8];
    end
    for (j = 0; j < 8; j = j + 1) begin : g_col_edge
      assign BV[0][j] = bq[j];
    end
  endgenerate

  // ---------------- the 8 x 8 grid ---------------------------------------
  generate
    for (i = 0; i < 8; i = i + 1) begin : g_pe_row
      for (j = 0; j < 8; j = j + 1) begin : g_pe_col
        mm_pe #(.J(j)) u_pe (
          .ap_clk  (ap_clk),
          .ap_rst_n(ap_rst_n),
          .a_in    (AH[i][j]),
          .a_f_in  (AF[i][j]),
          .a_l_in  (AL[i][j]),
          .a_v_in  (AV[i][j]),
          .a_out   (AH[i][j+1]),
          .a_f_out (AF[i][j+1]),
          .a_l_out (AL[i][j+1]),
          .a_v_out (AV[i][j+1]),
          .b_in    (BV[i][j]),
          .b_out   (BV[i+1][j]),
          .c_in    (CH[i][j]),
          .c_v_in  (CW[i][j]),
          .c_out   (CH[i][j+1]),
          .c_v_out (CW[i][j+1]));
      end
    end
  endgenerate

endmodule
