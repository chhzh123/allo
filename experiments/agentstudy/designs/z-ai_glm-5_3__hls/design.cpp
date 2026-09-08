// 8x8 output-stationary integer matrix-multiply tile.
//
// Architecture: 64 processing elements in an 8x8 grid, one dataflow process
// per element.  A[i][k] enters at PE(i,0) and travels east one neighbour per
// cycle; B[k][j] enters at PE(0,j) and travels south one neighbour per cycle.
// PE(i,j) multiplies the A value crossing it by the B value crossing it and
// accumulates C[i][j] in its own register (output stationary, one multiplier
// and one accumulator per element, no adder tree).
//
// Each element runs an 8-cycle loop.  While computing product p it drains
// product p-1 eastward along its row: at iteration t it forwards the value
// from its western neighbour for t < j and emits its own result at t == j,
// so output port i receives C[i][0]..C[i][7] in order, each value crossing
// only nearest neighbours.  Products therefore start every 8 cycles.
//
// The final product has no successor to drain it, so at the start of each
// product a processing element checks whether its A input has run dry.  When
// it has (no further product is being offered), the element drains the
// product it just computed along the row and then idles, waiting for inputs
// to appear again.

#include <hls_stream.h>
#include <ap_int.h>

template <int I, int J>
static void pe(hls::stream<ap_int<8> > &a_in, hls::stream<ap_int<8> > &a_out,
               hls::stream<ap_int<8> > &b_in, hls::stream<ap_int<8> > &b_out,
               hls::stream<ap_int<32> > &c_in, hls::stream<ap_int<32> > &c_out) {
#pragma HLS inline off
  ap_int<32> acc = 0;      // C[i][j] of the product being computed
  ap_int<32> acc_prev = 0; // C[i][j] of the previous product, being drained
  bool first = true;       // nothing to drain before the first product

  for (;;) {
    bool drain_now = false; // set when no further product is offered
    for (int t = 0; t < 8; t++) {
#pragma HLS pipeline II=1
      // At the first iteration of a product, a dry A input means there is no
      // further product: drain the result just computed instead.
      if (t == 0 && !first && a_in.empty()) drain_now = true;
      if (drain_now) {
        // Drain this row eastward: forward the j values from the west, then
        // emit this element's own result.
        if (t < J)       c_out.write(c_in.read());
        else if (t == J) c_out.write(acc);
      } else {
        ap_int<8> av = a_in.read();   // A[i][k], from the west
        if (J < 7) a_out.write(av);   // pass it east
        ap_int<8> bv = b_in.read();   // B[k][j], from the north
        if (I < 7) b_out.write(bv);   // pass it south
        ap_int<16> prod = av * bv;
#pragma HLS bind_op variable=prod op=mul impl=dsp
        acc += prod;
        // Drain the previous product's row results eastward, in order:
        // t < j  -> forward the value from the western neighbour,
        // t == j -> this element's own previous result.
        if (!first) {
          if (t < J)       c_out.write(c_in.read());
          else if (t == J) c_out.write(acc_prev);
        }
      }
    }
    if (drain_now) {
      acc = 0;    // the drained result is gone
      first = true; // a fresh product, when one arrives, has nothing to drain
    } else {
      acc_prev = acc;
      acc = 0;
      first = false;
    }
  }
}

// da/db/dc are placeholders for the edges of the grid that no element ever
// drives or reads (the writes/reads touching them are compiled out by the
// static I/J conditionals).
#define ROW0()                                                              \
  pe<0,0>(a_in_0, a_row[0][0], b_in_0, b_col[0][0], dc[0], c_row[0][0]); \
  pe<0,1>(a_row[0][0], a_row[0][1], b_in_1, b_col[0][1], c_row[0][0], c_row[0][1]); \
  pe<0,2>(a_row[0][1], a_row[0][2], b_in_2, b_col[0][2], c_row[0][1], c_row[0][2]); \
  pe<0,3>(a_row[0][2], a_row[0][3], b_in_3, b_col[0][3], c_row[0][2], c_row[0][3]); \
  pe<0,4>(a_row[0][3], a_row[0][4], b_in_4, b_col[0][4], c_row[0][3], c_row[0][4]); \
  pe<0,5>(a_row[0][4], a_row[0][5], b_in_5, b_col[0][5], c_row[0][4], c_row[0][5]); \
  pe<0,6>(a_row[0][5], a_row[0][6], b_in_6, b_col[0][6], c_row[0][5], c_row[0][6]); \
  pe<0,7>(a_row[0][6], da[0], b_in_7, b_col[0][7], c_row[0][6], c_out_0);

#define ROWI(I)                                                             \
  pe<I,0>(a_in_##I, a_row[I][0], b_col[I-1][0], b_col[I][0], dc[I], c_row[I][0]); \
  pe<I,1>(a_row[I][0], a_row[I][1], b_col[I-1][1], b_col[I][1], c_row[I][0], c_row[I][1]); \
  pe<I,2>(a_row[I][1], a_row[I][2], b_col[I-1][2], b_col[I][2], c_row[I][1], c_row[I][2]); \
  pe<I,3>(a_row[I][2], a_row[I][3], b_col[I-1][3], b_col[I][3], c_row[I][2], c_row[I][3]); \
  pe<I,4>(a_row[I][3], a_row[I][4], b_col[I-1][4], b_col[I][4], c_row[I][3], c_row[I][4]); \
  pe<I,5>(a_row[I][4], a_row[I][5], b_col[I-1][5], b_col[I][5], c_row[I][4], c_row[I][5]); \
  pe<I,6>(a_row[I][5], a_row[I][6], b_col[I-1][6], b_col[I][6], c_row[I][5], c_row[I][6]); \
  pe<I,7>(a_row[I][6], da[I], b_col[I-1][7], b_col[I][7], c_row[I][6], c_out_##I);

#define ROW7()                                                              \
  pe<7,0>(a_in_7, a_row[7][0], b_col[6][0], db[0], dc[7], c_row[7][0]); \
  pe<7,1>(a_row[7][0], a_row[7][1], b_col[6][1], db[1], c_row[7][0], c_row[7][1]); \
  pe<7,2>(a_row[7][1], a_row[7][2], b_col[6][2], db[2], c_row[7][1], c_row[7][2]); \
  pe<7,3>(a_row[7][2], a_row[7][3], b_col[6][3], db[3], c_row[7][2], c_row[7][3]); \
  pe<7,4>(a_row[7][3], a_row[7][4], b_col[6][4], db[4], c_row[7][3], c_row[7][4]); \
  pe<7,5>(a_row[7][4], a_row[7][5], b_col[6][5], db[5], c_row[7][4], c_row[7][5]); \
  pe<7,6>(a_row[7][5], a_row[7][6], b_col[6][6], db[6], c_row[7][5], c_row[7][6]); \
  pe<7,7>(a_row[7][6], da[7], b_col[6][7], db[7], c_row[7][6], c_out_7);

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

  // Nearest-neighbour links: a_row[i][j] carries A east from PE(i,j) to
  // PE(i,j+1), b_col[i][j] carries B south from PE(i,j) to PE(i+1,j), and
  // c_row[i][j] carries results east from PE(i,j) to PE(i,j+1).
  hls::stream<ap_int<8> >  a_row[8][7];
  hls::stream<ap_int<8> >  b_col[7][8];
  hls::stream<ap_int<32> > c_row[8][7];
  hls::stream<ap_int<8> >  da[8];
  hls::stream<ap_int<8> >  db[8];
  hls::stream<ap_int<32> > dc[8];
#pragma HLS stream variable=a_row depth=2
#pragma HLS stream variable=b_col depth=2
#pragma HLS stream variable=c_row depth=2
#pragma HLS stream variable=da depth=2
#pragma HLS stream variable=db depth=2
#pragma HLS stream variable=dc depth=2

  ROW0()
  ROWI(1)
  ROWI(2)
  ROWI(3)
  ROWI(4)
  ROWI(5)
  ROWI(6)
  ROW7()
}