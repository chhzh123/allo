// 8x8 output-stationary systolic matrix-multiply tile.
//
//   * 64 processing elements in an 8x8 grid, one process each.
//   * Each PE owns exactly one multiplier and one accumulator and keeps
//     C[i][j] until it is drained.
//   * A travels east  : a_in_i -> PE(i,0) -> PE(i,1) -> ... -> PE(i,7)
//   * B travels south : b_in_j -> PE(0,j) -> PE(1,j) -> ... -> PE(7,j)
//   * results travel east along each row: PE(i,0) -> ... -> PE(i,7) -> c_out_i
//     so port i emits C[i][0], C[i][1], ... C[i][7] in that order.
//   * Every link is a point to point channel between neighbours; nothing is
//     broadcast and no PE touches a port that is not on its own edge.
//
// Each PE runs one 8-cycle pass per product: in pass p it multiply-accumulates
// the eight k-terms of product p while at the same time shifting the results
// of product p-1 one step east.  Products therefore start every 8 cycles.

#include <hls_stream.h>
#include <ap_int.h>

typedef ap_int<8>  data_t;   // A / B element
typedef ap_int<16> prod_t;   // exact 8x8 product
typedef ap_int<32> acc_t;    // C element

// ---------------------------------------------------------------------------
// PE(i,0), i = 0..6 : A arrives from the input port, nothing arrives from the
// west, so this element only injects its own result into the row chain.
// ---------------------------------------------------------------------------
static void pe_w(hls::stream<data_t> &a_in, hls::stream<data_t> &a_out,
                 hls::stream<data_t> &b_in, hls::stream<data_t> &b_out,
                 hls::stream<acc_t> &c_out) {
#pragma HLS inline off
  acc_t hold = 0;              // the accumulator, held until drained
  bool  first = true;
  while (1) {
    acc_t acc = 0;
    for (int t = 0; t < 8; t++) {
#pragma HLS pipeline II=1
      data_t a = a_in.read();
      data_t b = b_in.read();
      a_out.write(a);          // pass A to the eastern neighbour
      b_out.write(b);          // pass B to the southern neighbour
      prod_t p = a * b;
#pragma HLS bind_op variable=p op=mul impl=dsp
      acc += p;
      if (!first && t == 0) c_out.write(hold);
    }
    hold  = acc;
    first = false;
  }
}

// ---------------------------------------------------------------------------
// PE(i,col), i = 0..6, col = 1..6 : interior element.
// ---------------------------------------------------------------------------
static void pe_m(int col,
                 hls::stream<data_t> &a_in, hls::stream<data_t> &a_out,
                 hls::stream<data_t> &b_in, hls::stream<data_t> &b_out,
                 hls::stream<acc_t> &c_in, hls::stream<acc_t> &c_out) {
#pragma HLS inline off
  acc_t hold = 0;
  bool  first = true;
  while (1) {
    acc_t acc = 0;
    for (int t = 0; t < 8; t++) {
#pragma HLS pipeline II=1
      data_t a = a_in.read();
      data_t b = b_in.read();
      a_out.write(a);
      b_out.write(b);
      prod_t p = a * b;
#pragma HLS bind_op variable=p op=mul impl=dsp
      acc += p;
      if (!first) {
        if (t < col)       c_out.write(c_in.read());  // relay from the west
        else if (t == col) c_out.write(hold);         // then its own result
      }
    }
    hold  = acc;
    first = false;
  }
}

// ---------------------------------------------------------------------------
// PE(i,7), i = 0..6 : eastern edge, drives output port i.
// ---------------------------------------------------------------------------
static void pe_e(hls::stream<data_t> &a_in,
                 hls::stream<data_t> &b_in, hls::stream<data_t> &b_out,
                 hls::stream<acc_t> &c_in, hls::stream<acc_t> &c_out) {
#pragma HLS inline off
  acc_t hold = 0;
  bool  first = true;
  while (1) {
    acc_t acc = 0;
    for (int t = 0; t < 8; t++) {
#pragma HLS pipeline II=1
      data_t a = a_in.read();
      data_t b = b_in.read();
      b_out.write(b);
      prod_t p = a * b;
#pragma HLS bind_op variable=p op=mul impl=dsp
      acc += p;
      if (!first) {
        if (t < 7) c_out.write(c_in.read());
        else       c_out.write(hold);
      }
    }
    hold  = acc;
    first = false;
  }
}

// ---------------------------------------------------------------------------
// PE(7,0) : southern-western corner, no southern neighbour.
// ---------------------------------------------------------------------------
static void pe_sw(hls::stream<data_t> &a_in, hls::stream<data_t> &a_out,
                  hls::stream<data_t> &b_in,
                  hls::stream<acc_t> &c_out) {
#pragma HLS inline off
  acc_t hold = 0;
  bool  first = true;
  while (1) {
    acc_t acc = 0;
    for (int t = 0; t < 8; t++) {
#pragma HLS pipeline II=1
      data_t a = a_in.read();
      data_t b = b_in.read();
      a_out.write(a);
      prod_t p = a * b;
#pragma HLS bind_op variable=p op=mul impl=dsp
      acc += p;
      if (!first && t == 0) c_out.write(hold);
    }
    hold  = acc;
    first = false;
  }
}

// ---------------------------------------------------------------------------
// PE(7,col), col = 1..6 : southern edge.
// ---------------------------------------------------------------------------
static void pe_s(int col,
                 hls::stream<data_t> &a_in, hls::stream<data_t> &a_out,
                 hls::stream<data_t> &b_in,
                 hls::stream<acc_t> &c_in, hls::stream<acc_t> &c_out) {
#pragma HLS inline off
  acc_t hold = 0;
  bool  first = true;
  while (1) {
    acc_t acc = 0;
    for (int t = 0; t < 8; t++) {
#pragma HLS pipeline II=1
      data_t a = a_in.read();
      data_t b = b_in.read();
      a_out.write(a);
      prod_t p = a * b;
#pragma HLS bind_op variable=p op=mul impl=dsp
      acc += p;
      if (!first) {
        if (t < col)       c_out.write(c_in.read());
        else if (t == col) c_out.write(hold);
      }
    }
    hold  = acc;
    first = false;
  }
}

// ---------------------------------------------------------------------------
// PE(7,7) : southern-eastern corner, drives output port 7.
// ---------------------------------------------------------------------------
static void pe_se(hls::stream<data_t> &a_in,
                  hls::stream<data_t> &b_in,
                  hls::stream<acc_t> &c_in, hls::stream<acc_t> &c_out) {
#pragma HLS inline off
  acc_t hold = 0;
  bool  first = true;
  while (1) {
    acc_t acc = 0;
    for (int t = 0; t < 8; t++) {
#pragma HLS pipeline II=1
      data_t a = a_in.read();
      data_t b = b_in.read();
      prod_t p = a * b;
#pragma HLS bind_op variable=p op=mul impl=dsp
      acc += p;
      if (!first) {
        if (t < 7) c_out.write(c_in.read());
        else       c_out.write(hold);
      }
    }
    hold  = acc;
    first = false;
  }
}

// ---------------------------------------------------------------------------
// The grid.
// ---------------------------------------------------------------------------
void gemm_tile(
    hls::stream<ap_int<8> > &a_in_0,
    hls::stream<ap_int<8> > &a_in_1,
    hls::stream<ap_int<8> > &a_in_2,
    hls::stream<ap_int<8> > &a_in_3,
    hls::stream<ap_int<8> > &a_in_4,
    hls::stream<ap_int<8> > &a_in_5,
    hls::stream<ap_int<8> > &a_in_6,
    hls::stream<ap_int<8> > &a_in_7,
    hls::stream<ap_int<8> > &b_in_0,
    hls::stream<ap_int<8> > &b_in_1,
    hls::stream<ap_int<8> > &b_in_2,
    hls::stream<ap_int<8> > &b_in_3,
    hls::stream<ap_int<8> > &b_in_4,
    hls::stream<ap_int<8> > &b_in_5,
    hls::stream<ap_int<8> > &b_in_6,
    hls::stream<ap_int<8> > &b_in_7,
    hls::stream<ap_int<32> > &c_out_0,
    hls::stream<ap_int<32> > &c_out_1,
    hls::stream<ap_int<32> > &c_out_2,
    hls::stream<ap_int<32> > &c_out_3,
    hls::stream<ap_int<32> > &c_out_4,
    hls::stream<ap_int<32> > &c_out_5,
    hls::stream<ap_int<32> > &c_out_6,
    hls::stream<ap_int<32> > &c_out_7) {
#pragma HLS interface ap_ctrl_none port=return
#pragma HLS dataflow

  // As[i][j] : PE(i,j) -> PE(i,j+1)   (A moving east)
  hls::stream<data_t> As[8][7];
#pragma HLS stream variable=As depth=2
  // Bs[i][j] : PE(i,j) -> PE(i+1,j)   (B moving south)
  hls::stream<data_t> Bs[7][8];
#pragma HLS stream variable=Bs depth=2
  // Cs[i][j] : PE(i,j) -> PE(i,j+1)   (results moving east)
  hls::stream<acc_t>  Cs[8][7];
#pragma HLS stream variable=Cs depth=2

  // ---- row 0 : B enters from the input ports ----
  pe_w(   a_in_0,    As[0][0], b_in_0, Bs[0][0],           Cs[0][0]);
  pe_m(1, As[0][0],  As[0][1], b_in_1, Bs[0][1], Cs[0][0], Cs[0][1]);
  pe_m(2, As[0][1],  As[0][2], b_in_2, Bs[0][2], Cs[0][1], Cs[0][2]);
  pe_m(3, As[0][2],  As[0][3], b_in_3, Bs[0][3], Cs[0][2], Cs[0][3]);
  pe_m(4, As[0][3],  As[0][4], b_in_4, Bs[0][4], Cs[0][3], Cs[0][4]);
  pe_m(5, As[0][4],  As[0][5], b_in_5, Bs[0][5], Cs[0][4], Cs[0][5]);
  pe_m(6, As[0][5],  As[0][6], b_in_6, Bs[0][6], Cs[0][5], Cs[0][6]);
  pe_e(   As[0][6],            b_in_7, Bs[0][7], Cs[0][6], c_out_0);

  // ---- row 1 ----
  pe_w(   a_in_1,    As[1][0], Bs[0][0], Bs[1][0],           Cs[1][0]);
  pe_m(1, As[1][0],  As[1][1], Bs[0][1], Bs[1][1], Cs[1][0], Cs[1][1]);
  pe_m(2, As[1][1],  As[1][2], Bs[0][2], Bs[1][2], Cs[1][1], Cs[1][2]);
  pe_m(3, As[1][2],  As[1][3], Bs[0][3], Bs[1][3], Cs[1][2], Cs[1][3]);
  pe_m(4, As[1][3],  As[1][4], Bs[0][4], Bs[1][4], Cs[1][3], Cs[1][4]);
  pe_m(5, As[1][4],  As[1][5], Bs[0][5], Bs[1][5], Cs[1][4], Cs[1][5]);
  pe_m(6, As[1][5],  As[1][6], Bs[0][6], Bs[1][6], Cs[1][5], Cs[1][6]);
  pe_e(   As[1][6],            Bs[0][7], Bs[1][7], Cs[1][6], c_out_1);

  // ---- row 2 ----
  pe_w(   a_in_2,    As[2][0], Bs[1][0], Bs[2][0],           Cs[2][0]);
  pe_m(1, As[2][0],  As[2][1], Bs[1][1], Bs[2][1], Cs[2][0], Cs[2][1]);
  pe_m(2, As[2][1],  As[2][2], Bs[1][2], Bs[2][2], Cs[2][1], Cs[2][2]);
  pe_m(3, As[2][2],  As[2][3], Bs[1][3], Bs[2][3], Cs[2][2], Cs[2][3]);
  pe_m(4, As[2][3],  As[2][4], Bs[1][4], Bs[2][4], Cs[2][3], Cs[2][4]);
  pe_m(5, As[2][4],  As[2][5], Bs[1][5], Bs[2][5], Cs[2][4], Cs[2][5]);
  pe_m(6, As[2][5],  As[2][6], Bs[1][6], Bs[2][6], Cs[2][5], Cs[2][6]);
  pe_e(   As[2][6],            Bs[1][7], Bs[2][7], Cs[2][6], c_out_2);

  // ---- row 3 ----
  pe_w(   a_in_3,    As[3][0], Bs[2][0], Bs[3][0],           Cs[3][0]);
  pe_m(1, As[3][0],  As[3][1], Bs[2][1], Bs[3][1], Cs[3][0], Cs[3][1]);
  pe_m(2, As[3][1],  As[3][2], Bs[2][2], Bs[3][2], Cs[3][1], Cs[3][2]);
  pe_m(3, As[3][2],  As[3][3], Bs[2][3], Bs[3][3], Cs[3][2], Cs[3][3]);
  pe_m(4, As[3][3],  As[3][4], Bs[2][4], Bs[3][4], Cs[3][3], Cs[3][4]);
  pe_m(5, As[3][4],  As[3][5], Bs[2][5], Bs[3][5], Cs[3][4], Cs[3][5]);
  pe_m(6, As[3][5],  As[3][6], Bs[2][6], Bs[3][6], Cs[3][5], Cs[3][6]);
  pe_e(   As[3][6],            Bs[2][7], Bs[3][7], Cs[3][6], c_out_3);

  // ---- row 4 ----
  pe_w(   a_in_4,    As[4][0], Bs[3][0], Bs[4][0],           Cs[4][0]);
  pe_m(1, As[4][0],  As[4][1], Bs[3][1], Bs[4][1], Cs[4][0], Cs[4][1]);
  pe_m(2, As[4][1],  As[4][2], Bs[3][2], Bs[4][2], Cs[4][1], Cs[4][2]);
  pe_m(3, As[4][2],  As[4][3], Bs[3][3], Bs[4][3], Cs[4][2], Cs[4][3]);
  pe_m(4, As[4][3],  As[4][4], Bs[3][4], Bs[4][4], Cs[4][3], Cs[4][4]);
  pe_m(5, As[4][4],  As[4][5], Bs[3][5], Bs[4][5], Cs[4][4], Cs[4][5]);
  pe_m(6, As[4][5],  As[4][6], Bs[3][6], Bs[4][6], Cs[4][5], Cs[4][6]);
  pe_e(   As[4][6],            Bs[3][7], Bs[4][7], Cs[4][6], c_out_4);

  // ---- row 5 ----
  pe_w(   a_in_5,    As[5][0], Bs[4][0], Bs[5][0],           Cs[5][0]);
  pe_m(1, As[5][0],  As[5][1], Bs[4][1], Bs[5][1], Cs[5][0], Cs[5][1]);
  pe_m(2, As[5][1],  As[5][2], Bs[4][2], Bs[5][2], Cs[5][1], Cs[5][2]);
  pe_m(3, As[5][2],  As[5][3], Bs[4][3], Bs[5][3], Cs[5][2], Cs[5][3]);
  pe_m(4, As[5][3],  As[5][4], Bs[4][4], Bs[5][4], Cs[5][3], Cs[5][4]);
  pe_m(5, As[5][4],  As[5][5], Bs[4][5], Bs[5][5], Cs[5][4], Cs[5][5]);
  pe_m(6, As[5][5],  As[5][6], Bs[4][6], Bs[5][6], Cs[5][5], Cs[5][6]);
  pe_e(   As[5][6],            Bs[4][7], Bs[5][7], Cs[5][6], c_out_5);

  // ---- row 6 ----
  pe_w(   a_in_6,    As[6][0], Bs[5][0], Bs[6][0],           Cs[6][0]);
  pe_m(1, As[6][0],  As[6][1], Bs[5][1], Bs[6][1], Cs[6][0], Cs[6][1]);
  pe_m(2, As[6][1],  As[6][2], Bs[5][2], Bs[6][2], Cs[6][1], Cs[6][2]);
  pe_m(3, As[6][2],  As[6][3], Bs[5][3], Bs[6][3], Cs[6][2], Cs[6][3]);
  pe_m(4, As[6][3],  As[6][4], Bs[5][4], Bs[6][4], Cs[6][3], Cs[6][4]);
  pe_m(5, As[6][4],  As[6][5], Bs[5][5], Bs[6][5], Cs[6][4], Cs[6][5]);
  pe_m(6, As[6][5],  As[6][6], Bs[5][6], Bs[6][6], Cs[6][5], Cs[6][6]);
  pe_e(   As[6][6],            Bs[5][7], Bs[6][7], Cs[6][6], c_out_6);

  // ---- row 7 : southern edge, no southern neighbour ----
  pe_sw(   a_in_7,   As[7][0], Bs[6][0],           Cs[7][0]);
  pe_s(1, As[7][0],  As[7][1], Bs[6][1], Cs[7][0], Cs[7][1]);
  pe_s(2, As[7][1],  As[7][2], Bs[6][2], Cs[7][1], Cs[7][2]);
  pe_s(3, As[7][2],  As[7][3], Bs[6][3], Cs[7][2], Cs[7][3]);
  pe_s(4, As[7][3],  As[7][4], Bs[6][4], Cs[7][3], Cs[7][4]);
  pe_s(5, As[7][4],  As[7][5], Bs[6][5], Cs[7][4], Cs[7][5]);
  pe_s(6, As[7][5],  As[7][6], Bs[6][6], Cs[7][5], Cs[7][6]);
  pe_se(  As[7][6],            Bs[6][7], Cs[7][6], c_out_7);
}
