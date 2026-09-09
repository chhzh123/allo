// Output-stationary 8x8 systolic GEMM tile.
//
// 64 PEs, one function instantiated per position inside a dataflow region.
// A flows east through point-to-point streams, B flows south, results drain
// east along each row (west values forwarded first, then the PE's own
// accumulator) so that PE(i,7) emits C[i][0..7] in order on c_out_i.
// Every PE runs a uniform 16-iteration cadence per product (8 accumulate,
// 1+J drain, the rest streamless padding) hence product interval <= 16.
#include <hls_stream.h>
#include <ap_int.h>

static void pe(const int I, const int J,
               hls::stream<ap_int<8> > &a_in,   // west: external (J==0) or link
               hls::stream<ap_int<8> > &b_in,   // north: external (I==0) or link
               hls::stream<ap_int<8> > &a_out,  // east link (unused when J==7)
               hls::stream<ap_int<8> > &b_out,  // south link (unused when I==7)
               hls::stream<ap_int<32> > &d_in,  // west drain link (unused J==0)
               hls::stream<ap_int<32> > &d_out) // east drain / c_out (J==7)
{
#pragma HLS inline off
  ap_int<32> acc = 0;
  ap_int<32> p;
#pragma HLS bind_op variable=p op=mul impl=dsp
  ap_uint<4> t = 0;      // 0..7 accumulate, 8..8+J drain, rest padding
  const int NDRAIN = (J == 0) ? 1 : (J + 1);
  while (true) {
#pragma HLS pipeline II=1
    bool advance = false;
    int ti = (int)t;
    if (ti < 8) {
      // accumulation phase: gate on data at west/north and room downstream
      if (!a_in.empty() && !b_in.empty()
          && (J == 7 || !a_out.full())
          && (I == 7 || !b_out.full())) {
        ap_int<8> a = a_in.read();
        ap_int<8> b = b_in.read();
        if (J < 7) a_out.write(a);  // A travels east
        if (I < 7) b_out.write(b);  // B travels south
        p = ap_int<16>(a) * ap_int<16>(b);
        acc += p;
        advance = true;
      }
    } else if (ti - 8 < NDRAIN) {
      // drain phase: forward west values first, then own accumulator
      bool own = (J == 0) || (ti - 8 >= J);
      if ((J == 0 || own || !d_in.empty()) && !d_out.full()) {
        ap_int<32> v = own ? acc : d_in.read();
        d_out.write(v);
        if (own) acc = 0;
        advance = true;
      }
    } else {
      advance = true;  // padding iteration: no stream traffic
    }
    if (advance) t = (ap_uint<4>)((ti == 15) ? 0 : ti + 1);
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

  hls::stream<ap_int<8> > a_e[56];     // A links between columns:  i*7+j, j=0..6
  hls::stream<ap_int<8> > b_s[56];     // B links between rows:     i*8+j, i=0..6
  hls::stream<ap_int<32> > d_e[56];    // drain links between columns
  hls::stream<ap_int<8> > sink8;       // discarded operands at east/south edges
  hls::stream<ap_int<32> > sink32;     // discarded drain input at west edge

#define PE(I, J)                                                           \
  pe((I), (J),                                                             \
     ((J) == 0 ? a_in_##I : a_e[(I) * 7 + (J) - 1]),                      \
     ((I) == 0 ? b_in_##J : b_s[((I) - 1) * 8 + (J)]),                    \
     ((J) == 7 ? sink8 : a_e[(I) * 7 + (J)]),                             \
     ((I) == 7 ? sink8 : b_s[(I) * 8 + (J)]),                             \
     ((J) == 0 ? sink32 : d_e[(I) * 7 + (J) - 1]),                        \
     ((J) == 7 ? c_out_##I : d_e[(I) * 7 + (J)]))

#define PE_ROW(I)                                                        \
  PE(I, 0); PE(I, 1); PE(I, 2); PE(I, 3);                                \
  PE(I, 4); PE(I, 5); PE(I, 6); PE(I, 7);

  PE_ROW(0)
  PE_ROW(1)
  PE_ROW(2)
  PE_ROW(3)
  PE_ROW(4)
  PE_ROW(5)
  PE_ROW(6)
  PE_ROW(7)

#undef PE
#undef PE_ROW
}
