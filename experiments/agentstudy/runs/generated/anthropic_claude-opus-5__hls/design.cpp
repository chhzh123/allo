// ---------------------------------------------------------------------------
// 8x8 output-stationary systolic matrix-multiply tile.
//
//   * 64 processing elements in an 8x8 grid, PE(i,j), i = row, j = column.
//   * PE(i,j) owns exactly one multiplier and exactly one accumulator; it
//     accumulates C[i][j] over all eight k and holds it in its own result
//     register until it is drained.  No partial sum of any C element exists
//     anywhere else and there is no adder tree over the grid.
//   * A travels east : a_in_i -> PE(i,0) -> PE(i,1) -> ... -> PE(i,7)
//   * B travels south: b_in_j -> PE(0,j) -> PE(1,j) -> ... -> PE(7,j)
//     every hop is a point-to-point link between two neighbours, one value
//     per cycle, nothing is broadcast.
//   * results leave along the rows: the result register of PE(i,j) feeds the
//     result link of PE(i,j+1), and only PE(i,7) drives c_out_i, so an
//     element reaches an output port only through its eastern neighbours.
//   * the only ports an element touches are the ones on its own edge:
//     PE(i,0) reads a_in_i, PE(0,j) reads b_in_j, PE(i,7) writes c_out_i.
//
// Each PE is two pieces of concurrent hardware, wired only to each other:
//
//     pe_mac_* : the multiplier and the accumulator.  Each cycle it takes A
//                from the west and B from the north, hands them to the east
//                and the south, and multiply-accumulates.  After the eighth
//                k it loads the finished C[i][j] into the PE's result reg.
//     pe_res_* : the result register of the PE and its one-cycle link east.
//                It first passes on the j results arriving from the west, one
//                hop per cycle, then its own, so the order on the row link is
//                C[i][0], C[i][1], ... and c_out_i comes out in order.
//
// PE(i,j) sees A[i][k] only after j eastward hops and B[k][j] only after i
// southward hops, so its last multiply is at cycle i+j+7 and C[7][7] cannot
// be drained before cycle 22.  Every stage moves one value per cycle, so
// products stream through back to back, eight cycles apart.
// ---------------------------------------------------------------------------

#include <hls_stream.h>
#include <ap_int.h>

typedef ap_int<8>  data_t;   // an element of A or B
typedef ap_int<16> prod_t;   // one exact 8x8 product
typedef ap_int<32> acc_t;    // an element of C

// ===========================================================================
// multiply-accumulate half of a PE: four variants, differing only in which
// neighbours exist (column 7 has no eastern one, row 7 no southern one).
// ===========================================================================

// interior: forwards A east, B south
static void pe_mac_es(hls::stream<data_t> &a_w, hls::stream<data_t> &a_e,
                      hls::stream<data_t> &b_n, hls::stream<data_t> &b_s,
                      hls::stream<acc_t> &res) {
#pragma HLS inline off
  while (1) {
    acc_t acc = 0;                      // this PE's accumulator
    for (int k = 0; k < 8; k++) {
#pragma HLS pipeline II=1
      data_t a = a_w.read();            // from the west
      data_t b = b_n.read();            // from the north
      a_e.write(a);                     // one step east
      b_s.write(b);                     // one step south
      prod_t p = a * b;                 // this PE's one multiplier
#pragma HLS bind_op variable=p op=mul impl=dsp
      acc += p;
      if (k == 7) res.write(acc);       // load the result register
    }
  }
}

// column 7: no eastern neighbour
static void pe_mac_s(hls::stream<data_t> &a_w,
                     hls::stream<data_t> &b_n, hls::stream<data_t> &b_s,
                     hls::stream<acc_t> &res) {
#pragma HLS inline off
  while (1) {
    acc_t acc = 0;
    for (int k = 0; k < 8; k++) {
#pragma HLS pipeline II=1
      data_t a = a_w.read();
      data_t b = b_n.read();
      b_s.write(b);
      prod_t p = a * b;
#pragma HLS bind_op variable=p op=mul impl=dsp
      acc += p;
      if (k == 7) res.write(acc);
    }
  }
}

// row 7: no southern neighbour
static void pe_mac_e(hls::stream<data_t> &a_w, hls::stream<data_t> &a_e,
                     hls::stream<data_t> &b_n,
                     hls::stream<acc_t> &res) {
#pragma HLS inline off
  while (1) {
    acc_t acc = 0;
    for (int k = 0; k < 8; k++) {
#pragma HLS pipeline II=1
      data_t a = a_w.read();
      data_t b = b_n.read();
      a_e.write(a);
      prod_t p = a * b;
#pragma HLS bind_op variable=p op=mul impl=dsp
      acc += p;
      if (k == 7) res.write(acc);
    }
  }
}

// PE(7,7): neither eastern nor southern neighbour
static void pe_mac_c(hls::stream<data_t> &a_w,
                     hls::stream<data_t> &b_n,
                     hls::stream<acc_t> &res) {
#pragma HLS inline off
  while (1) {
    acc_t acc = 0;
    for (int k = 0; k < 8; k++) {
#pragma HLS pipeline II=1
      data_t a = a_w.read();
      data_t b = b_n.read();
      prod_t p = a * b;
#pragma HLS bind_op variable=p op=mul impl=dsp
      acc += p;
      if (k == 7) res.write(acc);
    }
  }
}

// ===========================================================================
// result register / eastward result link of a PE.
// column 0 has no western neighbour, so it only injects its own value.
// ===========================================================================
static void pe_res_w(hls::stream<acc_t> &own, hls::stream<acc_t> &c_e) {
#pragma HLS inline off
  while (1) {
    for (int t = 0; t < 8; t++) {
#pragma HLS pipeline II=1
      if (t == 0) c_e.write(own.read());   // C[i][0] leaves first
    }
  }
}

static void pe_res(int col, hls::stream<acc_t> &own,
                   hls::stream<acc_t> &c_w, hls::stream<acc_t> &c_e) {
#pragma HLS inline off
  while (1) {
    for (int t = 0; t < 8; t++) {
#pragma HLS pipeline II=1
      if (t < col)        c_e.write(c_w.read());  // relay one hop east
      else if (t == col)  c_e.write(own.read());  // then this PE's own C
    }
  }
}

// ===========================================================================
// the grid
// ===========================================================================
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

  // A links: As[i][j] runs from PE(i,j) to PE(i,j+1)
  hls::stream<data_t> As[8][7];
#pragma HLS stream variable=As depth=2
  // B links: Bs[i][j] runs from PE(i,j) to PE(i+1,j)
  hls::stream<data_t> Bs[7][8];
#pragma HLS stream variable=Bs depth=2
  // inside PE(i,j): accumulator -> that PE's result register
  hls::stream<acc_t> Rs[8][8];
#pragma HLS stream variable=Rs depth=2
  // result links: Cs[i][j] runs from PE(i,j) to PE(i,j+1)
  hls::stream<acc_t> Cs[8][7];
#pragma HLS stream variable=Cs depth=2

  // ---------------- row 0 : B enters from the input ports ----------------
  pe_mac_es(a_in_0,   As[0][0], b_in_0,   Bs[0][0], Rs[0][0]);
  pe_mac_es(As[0][0], As[0][1], b_in_1,   Bs[0][1], Rs[0][1]);
  pe_mac_es(As[0][1], As[0][2], b_in_2,   Bs[0][2], Rs[0][2]);
  pe_mac_es(As[0][2], As[0][3], b_in_3,   Bs[0][3], Rs[0][3]);
  pe_mac_es(As[0][3], As[0][4], b_in_4,   Bs[0][4], Rs[0][4]);
  pe_mac_es(As[0][4], As[0][5], b_in_5,   Bs[0][5], Rs[0][5]);
  pe_mac_es(As[0][5], As[0][6], b_in_6,   Bs[0][6], Rs[0][6]);
  pe_mac_s (As[0][6],           b_in_7,   Bs[0][7], Rs[0][7]);

  pe_res_w  (Rs[0][0],           Cs[0][0]);
  pe_res (1, Rs[0][1], Cs[0][0], Cs[0][1]);
  pe_res (2, Rs[0][2], Cs[0][1], Cs[0][2]);
  pe_res (3, Rs[0][3], Cs[0][2], Cs[0][3]);
  pe_res (4, Rs[0][4], Cs[0][3], Cs[0][4]);
  pe_res (5, Rs[0][5], Cs[0][4], Cs[0][5]);
  pe_res (6, Rs[0][6], Cs[0][5], Cs[0][6]);
  pe_res (7, Rs[0][7], Cs[0][6], c_out_0);

  // ---------------- row 1 ----------------
  pe_mac_es(a_in_1,   As[1][0], Bs[0][0], Bs[1][0], Rs[1][0]);
  pe_mac_es(As[1][0], As[1][1], Bs[0][1], Bs[1][1], Rs[1][1]);
  pe_mac_es(As[1][1], As[1][2], Bs[0][2], Bs[1][2], Rs[1][2]);
  pe_mac_es(As[1][2], As[1][3], Bs[0][3], Bs[1][3], Rs[1][3]);
  pe_mac_es(As[1][3], As[1][4], Bs[0][4], Bs[1][4], Rs[1][4]);
  pe_mac_es(As[1][4], As[1][5], Bs[0][5], Bs[1][5], Rs[1][5]);
  pe_mac_es(As[1][5], As[1][6], Bs[0][6], Bs[1][6], Rs[1][6]);
  pe_mac_s (As[1][6],           Bs[0][7], Bs[1][7], Rs[1][7]);

  pe_res_w  (Rs[1][0],           Cs[1][0]);
  pe_res (1, Rs[1][1], Cs[1][0], Cs[1][1]);
  pe_res (2, Rs[1][2], Cs[1][1], Cs[1][2]);
  pe_res (3, Rs[1][3], Cs[1][2], Cs[1][3]);
  pe_res (4, Rs[1][4], Cs[1][3], Cs[1][4]);
  pe_res (5, Rs[1][5], Cs[1][4], Cs[1][5]);
  pe_res (6, Rs[1][6], Cs[1][5], Cs[1][6]);
  pe_res (7, Rs[1][7], Cs[1][6], c_out_1);

  // ---------------- row 2 ----------------
  pe_mac_es(a_in_2,   As[2][0], Bs[1][0], Bs[2][0], Rs[2][0]);
  pe_mac_es(As[2][0], As[2][1], Bs[1][1], Bs[2][1], Rs[2][1]);
  pe_mac_es(As[2][1], As[2][2], Bs[1][2], Bs[2][2], Rs[2][2]);
  pe_mac_es(As[2][2], As[2][3], Bs[1][3], Bs[2][3], Rs[2][3]);
  pe_mac_es(As[2][3], As[2][4], Bs[1][4], Bs[2][4], Rs[2][4]);
  pe_mac_es(As[2][4], As[2][5], Bs[1][5], Bs[2][5], Rs[2][5]);
  pe_mac_es(As[2][5], As[2][6], Bs[1][6], Bs[2][6], Rs[2][6]);
  pe_mac_s (As[2][6],           Bs[1][7], Bs[2][7], Rs[2][7]);

  pe_res_w  (Rs[2][0],           Cs[2][0]);
  pe_res (1, Rs[2][1], Cs[2][0], Cs[2][1]);
  pe_res (2, Rs[2][2], Cs[2][1], Cs[2][2]);
  pe_res (3, Rs[2][3], Cs[2][2], Cs[2][3]);
  pe_res (4, Rs[2][4], Cs[2][3], Cs[2][4]);
  pe_res (5, Rs[2][5], Cs[2][4], Cs[2][5]);
  pe_res (6, Rs[2][6], Cs[2][5], Cs[2][6]);
  pe_res (7, Rs[2][7], Cs[2][6], c_out_2);

  // ---------------- row 3 ----------------
  pe_mac_es(a_in_3,   As[3][0], Bs[2][0], Bs[3][0], Rs[3][0]);
  pe_mac_es(As[3][0], As[3][1], Bs[2][1], Bs[3][1], Rs[3][1]);
  pe_mac_es(As[3][1], As[3][2], Bs[2][2], Bs[3][2], Rs[3][2]);
  pe_mac_es(As[3][2], As[3][3], Bs[2][3], Bs[3][3], Rs[3][3]);
  pe_mac_es(As[3][3], As[3][4], Bs[2][4], Bs[3][4], Rs[3][4]);
  pe_mac_es(As[3][4], As[3][5], Bs[2][5], Bs[3][5], Rs[3][5]);
  pe_mac_es(As[3][5], As[3][6], Bs[2][6], Bs[3][6], Rs[3][6]);
  pe_mac_s (As[3][6],           Bs[2][7], Bs[3][7], Rs[3][7]);

  pe_res_w  (Rs[3][0],           Cs[3][0]);
  pe_res (1, Rs[3][1], Cs[3][0], Cs[3][1]);
  pe_res (2, Rs[3][2], Cs[3][1], Cs[3][2]);
  pe_res (3, Rs[3][3], Cs[3][2], Cs[3][3]);
  pe_res (4, Rs[3][4], Cs[3][3], Cs[3][4]);
  pe_res (5, Rs[3][5], Cs[3][4], Cs[3][5]);
  pe_res (6, Rs[3][6], Cs[3][5], Cs[3][6]);
  pe_res (7, Rs[3][7], Cs[3][6], c_out_3);

  // ---------------- row 4 ----------------
  pe_mac_es(a_in_4,   As[4][0], Bs[3][0], Bs[4][0], Rs[4][0]);
  pe_mac_es(As[4][0], As[4][1], Bs[3][1], Bs[4][1], Rs[4][1]);
  pe_mac_es(As[4][1], As[4][2], Bs[3][2], Bs[4][2], Rs[4][2]);
  pe_mac_es(As[4][2], As[4][3], Bs[3][3], Bs[4][3], Rs[4][3]);
  pe_mac_es(As[4][3], As[4][4], Bs[3][4], Bs[4][4], Rs[4][4]);
  pe_mac_es(As[4][4], As[4][5], Bs[3][5], Bs[4][5], Rs[4][5]);
  pe_mac_es(As[4][5], As[4][6], Bs[3][6], Bs[4][6], Rs[4][6]);
  pe_mac_s (As[4][6],           Bs[3][7], Bs[4][7], Rs[4][7]);

  pe_res_w  (Rs[4][0],           Cs[4][0]);
  pe_res (1, Rs[4][1], Cs[4][0], Cs[4][1]);
  pe_res (2, Rs[4][2], Cs[4][1], Cs[4][2]);
  pe_res (3, Rs[4][3], Cs[4][2], Cs[4][3]);
  pe_res (4, Rs[4][4], Cs[4][3], Cs[4][4]);
  pe_res (5, Rs[4][5], Cs[4][4], Cs[4][5]);
  pe_res (6, Rs[4][6], Cs[4][5], Cs[4][6]);
  pe_res (7, Rs[4][7], Cs[4][6], c_out_4);

  // ---------------- row 5 ----------------
  pe_mac_es(a_in_5,   As[5][0], Bs[4][0], Bs[5][0], Rs[5][0]);
  pe_mac_es(As[5][0], As[5][1], Bs[4][1], Bs[5][1], Rs[5][1]);
  pe_mac_es(As[5][1], As[5][2], Bs[4][2], Bs[5][2], Rs[5][2]);
  pe_mac_es(As[5][2], As[5][3], Bs[4][3], Bs[5][3], Rs[5][3]);
  pe_mac_es(As[5][3], As[5][4], Bs[4][4], Bs[5][4], Rs[5][4]);
  pe_mac_es(As[5][4], As[5][5], Bs[4][5], Bs[5][5], Rs[5][5]);
  pe_mac_es(As[5][5], As[5][6], Bs[4][6], Bs[5][6], Rs[5][6]);
  pe_mac_s (As[5][6],           Bs[4][7], Bs[5][7], Rs[5][7]);

  pe_res_w  (Rs[5][0],           Cs[5][0]);
  pe_res (1, Rs[5][1], Cs[5][0], Cs[5][1]);
  pe_res (2, Rs[5][2], Cs[5][1], Cs[5][2]);
  pe_res (3, Rs[5][3], Cs[5][2], Cs[5][3]);
  pe_res (4, Rs[5][4], Cs[5][3], Cs[5][4]);
  pe_res (5, Rs[5][5], Cs[5][4], Cs[5][5]);
  pe_res (6, Rs[5][6], Cs[5][5], Cs[5][6]);
  pe_res (7, Rs[5][7], Cs[5][6], c_out_5);

  // ---------------- row 6 ----------------
  pe_mac_es(a_in_6,   As[6][0], Bs[5][0], Bs[6][0], Rs[6][0]);
  pe_mac_es(As[6][0], As[6][1], Bs[5][1], Bs[6][1], Rs[6][1]);
  pe_mac_es(As[6][1], As[6][2], Bs[5][2], Bs[6][2], Rs[6][2]);
  pe_mac_es(As[6][2], As[6][3], Bs[5][3], Bs[6][3], Rs[6][3]);
  pe_mac_es(As[6][3], As[6][4], Bs[5][4], Bs[6][4], Rs[6][4]);
  pe_mac_es(As[6][4], As[6][5], Bs[5][5], Bs[6][5], Rs[6][5]);
  pe_mac_es(As[6][5], As[6][6], Bs[5][6], Bs[6][6], Rs[6][6]);
  pe_mac_s (As[6][6],           Bs[5][7], Bs[6][7], Rs[6][7]);

  pe_res_w  (Rs[6][0],           Cs[6][0]);
  pe_res (1, Rs[6][1], Cs[6][0], Cs[6][1]);
  pe_res (2, Rs[6][2], Cs[6][1], Cs[6][2]);
  pe_res (3, Rs[6][3], Cs[6][2], Cs[6][3]);
  pe_res (4, Rs[6][4], Cs[6][3], Cs[6][4]);
  pe_res (5, Rs[6][5], Cs[6][4], Cs[6][5]);
  pe_res (6, Rs[6][6], Cs[6][5], Cs[6][6]);
  pe_res (7, Rs[6][7], Cs[6][6], c_out_6);

  // ---------------- row 7 : southern edge ----------------
  pe_mac_e (a_in_7,   As[7][0], Bs[6][0], Rs[7][0]);
  pe_mac_e (As[7][0], As[7][1], Bs[6][1], Rs[7][1]);
  pe_mac_e (As[7][1], As[7][2], Bs[6][2], Rs[7][2]);
  pe_mac_e (As[7][2], As[7][3], Bs[6][3], Rs[7][3]);
  pe_mac_e (As[7][3], As[7][4], Bs[6][4], Rs[7][4]);
  pe_mac_e (As[7][4], As[7][5], Bs[6][5], Rs[7][5]);
  pe_mac_e (As[7][5], As[7][6], Bs[6][6], Rs[7][6]);
  pe_mac_c (As[7][6],           Bs[6][7], Rs[7][7]);

  pe_res_w  (Rs[7][0],           Cs[7][0]);
  pe_res (1, Rs[7][1], Cs[7][0], Cs[7][1]);
  pe_res (2, Rs[7][2], Cs[7][1], Cs[7][2]);
  pe_res (3, Rs[7][3], Cs[7][2], Cs[7][3]);
  pe_res (4, Rs[7][4], Cs[7][3], Cs[7][4]);
  pe_res (5, Rs[7][5], Cs[7][4], Cs[7][5]);
  pe_res (6, Rs[7][6], Cs[7][5], Cs[7][6]);
  pe_res (7, Rs[7][7], Cs[7][6], c_out_7);
}
