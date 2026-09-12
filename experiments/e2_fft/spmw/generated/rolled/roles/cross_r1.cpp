
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <hls_vector.h>
using namespace std;
void cross_r1_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2
) {	// L2
  l_S__r_0__r: for (int _r = 0; _r < 4352; _r++) {	// L3
    float v4[2];
    {
      hls::vector< float, 2 > _vec = v0.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v4[_iv0] = _vec[_iv0];
      }
    }	// L4
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = v4[_iv0];
      }
      v1.write(_vec);
    }	// L5
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = v4[_iv0];
      }
      v2.write(_vec);
    }	// L6
  }
}

/// This is top function.
void top(

) {	// L10
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v5;
  #pragma HLS stream variable=v5 depth=8	// L11
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v6;
  #pragma HLS stream variable=v6 depth=8	// L12
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v7;
  #pragma HLS stream variable=v7 depth=8	// L13
  cross_r1_0(v5, v6, v7);	// L14
}

