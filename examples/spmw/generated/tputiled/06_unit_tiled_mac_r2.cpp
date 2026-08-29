
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>
using namespace std;
void tiled_mac_r2_0(
  hls::stream< hls::vector< int8_t, 2 > >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int8_t >& v3
) {	// L2
  int8_t v4[2];
  {
    hls::vector< int8_t, 2 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v4[_iv0] = _vec[_iv0];
    }
  }	// L4
  l_S_m_0_m: for (int m = 0; m < 6; m++) {	// L5
    l_S_t_0_t: for (int t = 0; t < 2; t++) {	// L6
      int8_t v7 = v1.read();	// L7
      int8_t a;	// L8
      a = v7;	// L9
      int32_t p;	// L10
      p = 0;	// L11
      int8_t v10 = v4[t];	// L12
      int32_t v11 = v10;	// L13
      int32_t wt;	// L14
      wt = v11;	// L15
      int32_t v13 = p;	// L16
      int8_t v14 = a;	// L17
      int32_t v15 = wt;	// L18
      ap_int<40> v16 = v14;	// L19
      ap_int<40> v17 = v15;	// L20
      ap_int<40> v18 = v16 * v17;	// L21
      ap_int<41> v19 = v13;	// L22
      ap_int<41> v20 = v18;	// L23
      ap_int<41> v21 = v19 + v20;	// L24
      v2.write(v21);	// L25
      int8_t v22 = a;	// L26
      v3.write(v22);	// L27
    }
  }
}

/// This is top function.
void top(

) {	// L32
  #pragma HLS dataflow
  // Stream of vectors: each vector packs int8_t array[2] into hls::vector<int8_t, 2>
  hls::stream< hls::vector< int8_t, 2 > > v23;
  #pragma HLS stream variable=v23 depth=2	// L33
  hls::stream< int8_t > v24;
  #pragma HLS stream variable=v24 depth=2	// L34
  hls::stream< int8_t > v25;
  #pragma HLS stream variable=v25 depth=2	// L35
  hls::stream< int32_t > v26;
  #pragma HLS stream variable=v26 depth=2	// L36
  tiled_mac_r2_0(v23, v24, v26, v25);	// L37
}

