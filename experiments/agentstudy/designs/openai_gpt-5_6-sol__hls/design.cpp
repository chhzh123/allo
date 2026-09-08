// 8 by 8 output-stationary systolic matrix multiply.
#include <hls_stream.h>
#include <ap_int.h>

template<int D>
static void edge_delay(hls::stream<ap_int<8> > &in,
                       hls::stream<ap_int<8> > &out) {
#pragma HLS inline off
  ap_int<8> delay[D + 1];
#pragma HLS array_partition variable=delay complete
  ap_uint<4> filling = 0;
  while (1) {
#pragma HLS pipeline II=1
    ap_int<8> x = in.read();
    if (D == 0) {
      out.write(x);
    } else {
      ap_int<8> y = delay[D - 1];
      for (int n = D - 1; n > 0; --n) {
#pragma HLS unroll
        delay[n] = delay[n - 1];
      }
      delay[0] = x;
      if (filling == D)
        out.write(y);
      else
        filling = filling + 1;
    }
  }
}

// One PE: one multiplier, one output-stationary accumulator, and only the
// east/south operand links to adjacent PEs.  A completed accumulator enters a
// private one-writer/one-reader channel belonging to this PE's row drain.
static void mac_pe(hls::stream<ap_int<8> > &a_in,
                   hls::stream<ap_int<8> > &a_east,
                   hls::stream<ap_int<8> > &b_in,
                   hls::stream<ap_int<8> > &b_south,
                   hls::stream<ap_int<32> > &completed) {
#pragma HLS inline off
  ap_int<32> accum = 0;
  ap_uint<3> k = 0;
  while (1) {
#pragma HLS pipeline II=1
    ap_int<8> av = a_in.read();
    ap_int<8> bv = b_in.read();
    a_east.write(av);
    b_south.write(bv);

    ap_int<32> product = av * bv;
#pragma HLS bind_op variable=product op=mul impl=dsp
    ap_int<32> next = (k == 0) ? product : (ap_int<32>)(accum + product);
    accum = next;
    if (k == 7)
      completed.write(next);
    k = k + 1;
  }
}

// Result movement is a nearest-neighbour chain along each row.  For each
// product a PE first sends its stationary result, then relays all results to
// its east.  Inductively, PE zero emits columns 0 through 7 in order.
template<int EAST_COUNT>
static void drain_pe(hls::stream<ap_int<32> > &own,
                     hls::stream<ap_int<32> > &east,
                     hls::stream<ap_int<32> > &west) {
#pragma HLS inline off
  while (1) {
    west.write(own.read());
    for (int n = 0; n < EAST_COUNT; ++n) {
#pragma HLS pipeline II=1
      west.write(east.read());
    }
  }
}

static void drain_last(hls::stream<ap_int<32> > &own,
                       hls::stream<ap_int<32> > &west) {
#pragma HLS inline off
  while (1) {
#pragma HLS pipeline II=1
    west.write(own.read());
  }
}

static void discard_operand(hls::stream<ap_int<8> > &in) {
#pragma HLS inline off
  while (1) {
#pragma HLS pipeline II=1
    (void)in.read();
  }
}

static void output_link(hls::stream<ap_int<32> > &in,
                        hls::stream<ap_int<32> > &out) {
#pragma HLS inline off
  while (1) {
#pragma HLS pipeline II=1
    out.write(in.read());
  }
}

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

  hls::stream<ap_int<8> > a_link[8][9];
  hls::stream<ap_int<8> > b_link[9][8];
  hls::stream<ap_int<32> > done[8][8];
  hls::stream<ap_int<32> > result[8][8];
#pragma HLS stream variable=a_link depth=2
#pragma HLS stream variable=b_link depth=2
#pragma HLS stream variable=done depth=2
#pragma HLS stream variable=result depth=2

  edge_delay<0>(a_in_0, a_link[0][0]);
  edge_delay<1>(a_in_1, a_link[1][0]);
  edge_delay<2>(a_in_2, a_link[2][0]);
  edge_delay<3>(a_in_3, a_link[3][0]);
  edge_delay<4>(a_in_4, a_link[4][0]);
  edge_delay<5>(a_in_5, a_link[5][0]);
  edge_delay<6>(a_in_6, a_link[6][0]);
  edge_delay<7>(a_in_7, a_link[7][0]);
  edge_delay<0>(b_in_0, b_link[0][0]);
  edge_delay<1>(b_in_1, b_link[0][1]);
  edge_delay<2>(b_in_2, b_link[0][2]);
  edge_delay<3>(b_in_3, b_link[0][3]);
  edge_delay<4>(b_in_4, b_link[0][4]);
  edge_delay<5>(b_in_5, b_link[0][5]);
  edge_delay<6>(b_in_6, b_link[0][6]);
  edge_delay<7>(b_in_7, b_link[0][7]);

#define MAC_ROW(I) \
  mac_pe(a_link[I][0], a_link[I][1], b_link[I][0], b_link[I+1][0], done[I][0]); \
  mac_pe(a_link[I][1], a_link[I][2], b_link[I][1], b_link[I+1][1], done[I][1]); \
  mac_pe(a_link[I][2], a_link[I][3], b_link[I][2], b_link[I+1][2], done[I][2]); \
  mac_pe(a_link[I][3], a_link[I][4], b_link[I][3], b_link[I+1][3], done[I][3]); \
  mac_pe(a_link[I][4], a_link[I][5], b_link[I][4], b_link[I+1][4], done[I][4]); \
  mac_pe(a_link[I][5], a_link[I][6], b_link[I][5], b_link[I+1][5], done[I][5]); \
  mac_pe(a_link[I][6], a_link[I][7], b_link[I][6], b_link[I+1][6], done[I][6]); \
  mac_pe(a_link[I][7], a_link[I][8], b_link[I][7], b_link[I+1][7], done[I][7]);
  MAC_ROW(0) MAC_ROW(1) MAC_ROW(2) MAC_ROW(3)
  MAC_ROW(4) MAC_ROW(5) MAC_ROW(6) MAC_ROW(7)
#undef MAC_ROW

#define DRAIN_ROW(I) \
  drain_pe<7>(done[I][0], result[I][1], result[I][0]); \
  drain_pe<6>(done[I][1], result[I][2], result[I][1]); \
  drain_pe<5>(done[I][2], result[I][3], result[I][2]); \
  drain_pe<4>(done[I][3], result[I][4], result[I][3]); \
  drain_pe<3>(done[I][4], result[I][5], result[I][4]); \
  drain_pe<2>(done[I][5], result[I][6], result[I][5]); \
  drain_pe<1>(done[I][6], result[I][7], result[I][6]); \
  drain_last(done[I][7], result[I][7]);
  DRAIN_ROW(0) DRAIN_ROW(1) DRAIN_ROW(2) DRAIN_ROW(3)
  DRAIN_ROW(4) DRAIN_ROW(5) DRAIN_ROW(6) DRAIN_ROW(7)
#undef DRAIN_ROW

  discard_operand(a_link[0][8]); discard_operand(a_link[1][8]);
  discard_operand(a_link[2][8]); discard_operand(a_link[3][8]);
  discard_operand(a_link[4][8]); discard_operand(a_link[5][8]);
  discard_operand(a_link[6][8]); discard_operand(a_link[7][8]);
  discard_operand(b_link[8][0]); discard_operand(b_link[8][1]);
  discard_operand(b_link[8][2]); discard_operand(b_link[8][3]);
  discard_operand(b_link[8][4]); discard_operand(b_link[8][5]);
  discard_operand(b_link[8][6]); discard_operand(b_link[8][7]);

  output_link(result[0][0], c_out_0);
  output_link(result[1][0], c_out_1);
  output_link(result[2][0], c_out_2);
  output_link(result[3][0], c_out_3);
  output_link(result[4][0], c_out_4);
  output_link(result[5][0], c_out_5);
  output_link(result[6][0], c_out_6);
  output_link(result[7][0], c_out_7);
}
