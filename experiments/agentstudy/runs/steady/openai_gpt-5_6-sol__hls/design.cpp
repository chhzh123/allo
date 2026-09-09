// 8x8 output-stationary systolic integer GEMM tile.
#include <hls_stream.h>
#include <ap_int.h>

// Each PE owns exactly one multiplier and one accumulator.  A advances east
// and B south through registered neighbour streams.  The validity bits create
// the conventional wavefront skew without delaying any external input port.
static void pe(hls::stream<ap_int<8> > &a_in,
               hls::stream<ap_int<8> > &a_east,
               hls::stream<ap_uint<1> > &av_in,
               hls::stream<ap_uint<1> > &av_east,
               hls::stream<ap_int<8> > &b_in,
               hls::stream<ap_int<8> > &b_south,
               hls::stream<ap_uint<1> > &bv_in,
               hls::stream<ap_uint<1> > &bv_south,
               hls::stream<ap_int<32> > &done) {
#pragma HLS inline off
  ap_int<32> acc = 0;
  ap_uint<3> k = 0;
  while (1) {
#pragma HLS pipeline II=1
    ap_int<8> a = a_in.read();
    ap_int<8> b = b_in.read();
    ap_uint<1> va = av_in.read();
    ap_uint<1> vb = bv_in.read();
    a_east.write(a); av_east.write(va);
    b_south.write(b); bv_south.write(vb);
    if (va && vb) {
      ap_int<32> product = a * b;
#pragma HLS bind_op variable=product op=mul impl=dsp
      ap_int<32> next = (k == 0) ? product : (ap_int<32>)(acc + product);
      acc = next;
      if (k == 7) done.write(next);
      k = k + 1;
    }
  }
}

// Feed one real operand every clock and independently generate a validity
// pattern: eight valid clocks followed by D bubbles.  Thus product starts are
// 8+D clocks apart (no more than 15), while ports themselves can accept the
// next product after the mandatory eight clocks.
template<int D>
static void edge(hls::stream<ap_int<8> > &in,
                 hls::stream<ap_int<8> > &data,
                 hls::stream<ap_uint<1> > &valid) {
#pragma HLS inline off
  ap_uint<4> phase = 0;
  while (1) {
#pragma HLS pipeline II=1
    data.write(in.read());
    valid.write(phase < 8);
    phase = (phase == 7 + D) ? (ap_uint<4>)0 : (ap_uint<4>)(phase + 1);
  }
}

template<int N>
static void result_stage(hls::stream<ap_int<32> > &west,
                         hls::stream<ap_int<32> > &own,
                         hls::stream<ap_int<32> > &east) {
#pragma HLS inline off
  ap_uint<4> phase = 0;
  while (1) {
#pragma HLS pipeline II=1
    east.write(phase < N ? west.read() : own.read());
    phase = (phase == N) ? (ap_uint<4>)0 : (ap_uint<4>)(phase + 1);
  }
}
static void first(hls::stream<ap_int<32> > &own,hls::stream<ap_int<32> > &out) {
#pragma HLS inline off
  while(1) { #pragma HLS pipeline II=1
    out.write(own.read());
  }
}
static void sink8(hls::stream<ap_int<8> > &x) {
#pragma HLS inline off
  while(1) { #pragma HLS pipeline II=1
    (void)x.read();
  }
}
static void sink1(hls::stream<ap_uint<1> > &x) {
#pragma HLS inline off
  while(1) { #pragma HLS pipeline II=1
    (void)x.read();
  }
}
static void output(hls::stream<ap_int<32> > &x,hls::stream<ap_int<32> > &y) {
#pragma HLS inline off
  while(1) { #pragma HLS pipeline II=1
    y.write(x.read());
  }
}

void gemm_tile(
 hls::stream<ap_int<8> >&a_in_0,hls::stream<ap_int<8> >&a_in_1,hls::stream<ap_int<8> >&a_in_2,hls::stream<ap_int<8> >&a_in_3,
 hls::stream<ap_int<8> >&a_in_4,hls::stream<ap_int<8> >&a_in_5,hls::stream<ap_int<8> >&a_in_6,hls::stream<ap_int<8> >&a_in_7,
 hls::stream<ap_int<8> >&b_in_0,hls::stream<ap_int<8> >&b_in_1,hls::stream<ap_int<8> >&b_in_2,hls::stream<ap_int<8> >&b_in_3,
 hls::stream<ap_int<8> >&b_in_4,hls::stream<ap_int<8> >&b_in_5,hls::stream<ap_int<8> >&b_in_6,hls::stream<ap_int<8> >&b_in_7,
 hls::stream<ap_int<32> >&c_out_0,hls::stream<ap_int<32> >&c_out_1,hls::stream<ap_int<32> >&c_out_2,hls::stream<ap_int<32> >&c_out_3,
 hls::stream<ap_int<32> >&c_out_4,hls::stream<ap_int<32> >&c_out_5,hls::stream<ap_int<32> >&c_out_6,hls::stream<ap_int<32> >&c_out_7) {
#pragma HLS interface ap_ctrl_none port=return
#pragma HLS dataflow
 hls::stream<ap_int<8> > a[8][9],b[9][8];
 hls::stream<ap_uint<1> > av[8][9],bv[9][8];
 hls::stream<ap_int<32> > d[8][8],r[8][8];
#pragma HLS stream variable=a depth=2
#pragma HLS stream variable=b depth=2
#pragma HLS stream variable=av depth=2
#pragma HLS stream variable=bv depth=2
#pragma HLS stream variable=d depth=2
#pragma HLS stream variable=r depth=2
 edge<0>(a_in_0,a[0][0],av[0][0]); edge<1>(a_in_1,a[1][0],av[1][0]);
 edge<2>(a_in_2,a[2][0],av[2][0]); edge<3>(a_in_3,a[3][0],av[3][0]);
 edge<4>(a_in_4,a[4][0],av[4][0]); edge<5>(a_in_5,a[5][0],av[5][0]);
 edge<6>(a_in_6,a[6][0],av[6][0]); edge<7>(a_in_7,a[7][0],av[7][0]);
 edge<0>(b_in_0,b[0][0],bv[0][0]); edge<1>(b_in_1,b[0][1],bv[0][1]);
 edge<2>(b_in_2,b[0][2],bv[0][2]); edge<3>(b_in_3,b[0][3],bv[0][3]);
 edge<4>(b_in_4,b[0][4],bv[0][4]); edge<5>(b_in_5,b[0][5],bv[0][5]);
 edge<6>(b_in_6,b[0][6],bv[0][6]); edge<7>(b_in_7,b[0][7],bv[0][7]);
#define PR(I,J) pe(a[I][J],a[I][J+1],av[I][J],av[I][J+1],b[I][J],b[I+1][J],bv[I][J],bv[I+1][J],d[I][J]);
#define ROW(I) PR(I,0) PR(I,1) PR(I,2) PR(I,3) PR(I,4) PR(I,5) PR(I,6) PR(I,7)
 ROW(0) ROW(1) ROW(2) ROW(3) ROW(4) ROW(5) ROW(6) ROW(7)
#undef ROW
#undef PR
#define RR(I) first(d[I][0],r[I][0]); result_stage<1>(r[I][0],d[I][1],r[I][1]); result_stage<2>(r[I][1],d[I][2],r[I][2]); result_stage<3>(r[I][2],d[I][3],r[I][3]); result_stage<4>(r[I][3],d[I][4],r[I][4]); result_stage<5>(r[I][4],d[I][5],r[I][5]); result_stage<6>(r[I][5],d[I][6],r[I][6]); result_stage<7>(r[I][6],d[I][7],r[I][7]);
 RR(0) RR(1) RR(2) RR(3) RR(4) RR(5) RR(6) RR(7)
#undef RR
 sink8(a[0][8]);sink8(a[1][8]);sink8(a[2][8]);sink8(a[3][8]);sink8(a[4][8]);sink8(a[5][8]);sink8(a[6][8]);sink8(a[7][8]);
 sink1(av[0][8]);sink1(av[1][8]);sink1(av[2][8]);sink1(av[3][8]);sink1(av[4][8]);sink1(av[5][8]);sink1(av[6][8]);sink1(av[7][8]);
 sink8(b[8][0]);sink8(b[8][1]);sink8(b[8][2]);sink8(b[8][3]);sink8(b[8][4]);sink8(b[8][5]);sink8(b[8][6]);sink8(b[8][7]);
 sink1(bv[8][0]);sink1(bv[8][1]);sink1(bv[8][2]);sink1(bv[8][3]);sink1(bv[8][4]);sink1(bv[8][5]);sink1(bv[8][6]);sink1(bv[8][7]);
 output(r[0][7],c_out_0);output(r[1][7],c_out_1);output(r[2][7],c_out_2);output(r[3][7],c_out_3);
 output(r[4][7],c_out_4);output(r[5][7],c_out_5);output(r[6][7],c_out_6);output(r[7][7],c_out_7);
}
