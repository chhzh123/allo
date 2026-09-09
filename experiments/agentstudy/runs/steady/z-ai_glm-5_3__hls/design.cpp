// 8x8 output-stationary integer matrix-multiply tile.
//
// Architecture: 64 processing elements in an 8x8 grid, one dataflow process
// per element.  A[i][k] enters at PE(i,0) and travels east one neighbour per
// cycle; B[k][j] enters at PE(0,j) and travels south one neighbour per cycle.
// PE(i,j) multiplies the A value crossing it by the B value crossing it and
// accumulates C[i][j] in its own register (output stationary, one multiplier
// and one accumulator per element, no adder tree).
//
// Each element runs one 8-iteration pipelined loop per product.  While
// computing product p it drains product p-1 eastward along its row: at
// iteration t it forwards the value from its western neighbour for t < j and
// emits its own previous result at t == j, so output port i receives
// C[i][0]..C[i][7] in order, each value crossing only nearest neighbours.
// The internal result links carry one word every iteration (padding when a
// column has nothing real to send), so every link read is unconditional.
//
// The final product has no successor to drain it.  Only PE(i,0) can see the
// end directly, because its A input is a top-level port: once the last value
// of a product has been taken the port stays empty, whereas the links
// between elements always run dry between two products.  So at iteration 0
// of each product PE(i,0) checks its port; if no further product is offered
// it declares the end, and either way the A word it passes east carries a
// valid flag: 1 for a data value, 0 for the end marker.  The marker travels
// east one neighbour per cycle, exactly in step with the data wave, so it
// reaches each column at its own product boundary and that column spends
// the loop draining the product it just computed along its row.  On the
// next boundary PE(i,0) blocks waiting for fresh data, so no marker is ever
// sent twice.

#include <hls_stream.h>
#include <ap_int.h>

// Column 0: A input is the top-level 8-bit port; the end decision is made
// here from the port's empty status and rides east in the A word's valid bit.
template <int I>
static void pe0(hls::stream<ap_int<8> > &a_in, hls::stream<ap_uint<9> > &a_out,
                hls::stream<ap_int<8> > &b_in, hls::stream<ap_int<8> > &b_out,
                hls::stream<ap_int<32> > &c_out) {
#pragma HLS inline off
  ap_int<32> acc = 0;       // C[i][0] of the product being computed
  ap_int<32> acc_prev = 0;  // C[i][0] of the previous product, being drained
  bool first = true;        // nothing to drain before the first product
  bool idle = false;        // after a final drain, wait for a fresh product

  for (;;) {
    bool drain_now = false;
    ap_int<8> av = 0;
    for (int t = 0; t < 8; t++) {
#pragma HLS pipeline II=1
      if (t == 0) {
        if (idle) av = a_in.read();               // wait for the next product's first value
        else if (a_in.empty()) drain_now = true;  // no further product on offer
        else av = a_in.read();
      } else if (!drain_now) {
        av = a_in.read();                         // further A values of this product
      }
      // This element's own result goes first along the row (t == 0);
      // later iterations send padding, which never reaches an output port.
      ap_int<32> ov = (t == 0) ? acc_prev : (ap_int<32>)0;
      if (drain_now) {
        if (t == 0) a_out.write(0);               // pass the end marker east
      } else {
        ap_uint<9> w = (ap_uint<9>)0x100 | (ap_uint<8>)av;
        a_out.write(w);                           // pass A east, marked valid
        ap_int<8> bv = b_in.read();               // B[k][0], from the north
        if (I < 7) b_out.write(bv);               // pass it south
        ap_int<16> prod = av * bv;
#pragma HLS bind_op variable=prod op=mul impl=dsp
        acc += prod;
      }
      c_out.write(ov);                            // the result link always flows
    }
    if (drain_now) {
      first = true;
      idle = true;
      acc_prev = 0;
    } else {
      acc_prev = acc;
      acc = 0;
      first = false;
      idle = false;
    }
  }
}

// Columns 1..7: A arrives over an internal link whose words carry a valid
// flag, so the end decision needs no empty-status probe here -- the marker
// arrives exactly at this element's product boundary.
template <int I, int J>
static void pej(hls::stream<ap_uint<9> > &a_in, hls::stream<ap_uint<9> > &a_out,
                hls::stream<ap_int<8> > &b_in, hls::stream<ap_int<8> > &b_out,
                hls::stream<ap_int<32> > &c_in, hls::stream<ap_int<32> > &c_out) {
#pragma HLS inline off
  ap_int<32> acc = 0;       // C[i][j] of the product being computed
  ap_int<32> acc_prev = 0;  // C[i][j] of the previous product, being drained
  bool first = true;        // nothing to drain before the first product

  for (;;) {
    bool drain_now = false;
    ap_uint<9> w = 0;
    for (int t = 0; t < 8; t++) {
#pragma HLS pipeline II=1
      if (t == 0) {
        w = a_in.read();                          // first A word, or the end marker
        if (!w[8]) drain_now = true;
      } else if (!drain_now) {
        w = a_in.read();                          // further A values of this product
      }
      ap_int<32> cv = c_in.read();                // result arriving from the west
      ap_int<8> av = (ap_int<8>)w.range(7, 0);
      // At iteration t the row's t-th result passes eastward: forwarded from
      // the west for t < j, this element's own previous result at t == j,
      // padding afterwards (which never reaches an output port).
      ap_int<32> ov = (t < J) ? cv : ((t == J) ? acc_prev : cv);
      if (drain_now) {
        if (t == 0 && J < 7) a_out.write(w);      // pass the end marker east
      } else {
        if (J < 7) a_out.write(w);                // pass A east
        ap_int<8> bv = b_in.read();               // B[k][j], from the north
        if (I < 7) b_out.write(bv);               // pass it south
        ap_int<16> prod = av * bv;
#pragma HLS bind_op variable=prod op=mul impl=dsp
        acc += prod;
      }
      if (J < 7) c_out.write(ov);                 // internal link: always flows
      else if (!first) c_out.write(ov);           // output port: skip the first product
    }
    if (drain_now) {
      first = true;
      acc_prev = 0;
    } else {
      acc_prev = acc;
      acc = 0;
      first = false;
    }
  }
}

// da/db are placeholders for the edges of the grid that no element ever
// drives or reads (the writes touching them are compiled out by the static
// I/J conditionals).
#define ROW0()                                                              \
  pe0<0>(a_in_0, a_row[0][0], b_in_0, b_col[0][0], c_row[0][0]); \
  pej<0,1>(a_row[0][0], a_row[0][1], b_in_1, b_col[0][1], c_row[0][0], c_row[0][1]); \
  pej<0,2>(a_row[0][1], a_row[0][2], b_in_2, b_col[0][2], c_row[0][1], c_row[0][2]); \
  pej<0,3>(a_row[0][2], a_row[0][3], b_in_3, b_col[0][3], c_row[0][2], c_row[0][3]); \
  pej<0,4>(a_row[0][3], a_row[0][4], b_in_4, b_col[0][4], c_row[0][3], c_row[0][4]); \
  pej<0,5>(a_row[0][4], a_row[0][5], b_in_5, b_col[0][5], c_row[0][4], c_row[0][5]); \
  pej<0,6>(a_row[0][5], a_row[0][6], b_in_6, b_col[0][6], c_row[0][5], c_row[0][6]); \
  pej<0,7>(a_row[0][6], da[0], b_in_7, b_col[0][7], c_row[0][6], c_out_0);

#define ROWI(I)                                                             \
  pe0<I>(a_in_##I, a_row[I][0], b_col[I-1][0], b_col[I][0], c_row[I][0]); \
  pej<I,1>(a_row[I][0], a_row[I][1], b_col[I-1][1], b_col[I][1], c_row[I][0], c_row[I][1]); \
  pej<I,2>(a_row[I][1], a_row[I][2], b_col[I-1][2], b_col[I][2], c_row[I][1], c_row[I][2]); \
  pej<I,3>(a_row[I][2], a_row[I][3], b_col[I-1][3], b_col[I][3], c_row[I][2], c_row[I][3]); \
  pej<I,4>(a_row[I][3], a_row[I][4], b_col[I-1][4], b_col[I][4], c_row[I][3], c_row[I][4]); \
  pej<I,5>(a_row[I][4], a_row[I][5], b_col[I-1][5], b_col[I][5], c_row[I][4], c_row[I][5]); \
  pej<I,6>(a_row[I][5], a_row[I][6], b_col[I-1][6], b_col[I][6], c_row[I][5], c_row[I][6]); \
  pej<I,7>(a_row[I][6], da[I], b_col[I-1][7], b_col[I][7], c_row[I][6], c_out_##I);

#define ROW7()                                                              \
  pe0<7>(a_in_7, a_row[7][0], b_col[6][0], db[0], c_row[7][0]); \
  pej<7,1>(a_row[7][0], a_row[7][1], b_col[6][1], db[1], c_row[7][0], c_row[7][1]); \
  pej<7,2>(a_row[7][1], a_row[7][2], b_col[6][2], db[2], c_row[7][1], c_row[7][2]); \
  pej<7,3>(a_row[7][2], a_row[7][3], b_col[6][3], db[3], c_row[7][2], c_row[7][3]); \
  pej<7,4>(a_row[7][3], a_row[7][4], b_col[6][4], db[4], c_row[7][3], c_row[7][4]); \
  pej<7,5>(a_row[7][4], a_row[7][5], b_col[6][5], db[5], c_row[7][4], c_row[7][5]); \
  pej<7,6>(a_row[7][5], a_row[7][6], b_col[6][6], db[6], c_row[7][5], c_row[7][6]); \
  pej<7,7>(a_row[7][6], da[7], b_col[6][7], db[7], c_row[7][6], c_out_7);

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

  // Nearest-neighbour links: a_row[i][j] carries the 9-bit A word (valid flag
  // plus value) east from PE(i,j) to PE(i,j+1), b_col[i][j] carries B south
  // from PE(i,j) to PE(i+1,j), and c_row[i][j] carries results east from
  // PE(i,j) to PE(i,j+1).
  hls::stream<ap_uint<9> > a_row[8][7];
  hls::stream<ap_int<8> >  b_col[7][8];
  hls::stream<ap_int<32> > c_row[8][7];
  hls::stream<ap_uint<9> > da[8];
  hls::stream<ap_int<8> >  db[8];
#pragma HLS stream variable=a_row depth=2
#pragma HLS stream variable=b_col depth=2
#pragma HLS stream variable=c_row depth=2
#pragma HLS stream variable=da depth=2
#pragma HLS stream variable=db depth=2

  ROW0()
  ROWI(1)
  ROWI(2)
  ROWI(3)
  ROWI(4)
  ROWI(5)
  ROWI(6)
  ROW7()
}