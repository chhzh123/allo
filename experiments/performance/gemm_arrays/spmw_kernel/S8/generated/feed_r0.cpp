
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>
using namespace std;
void feed_r0_0(
  hls::stream< int32_t >& v0,
  hls::stream< hls::vector< int8_t, 8 > >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< hls::vector< int8_t, 8 > >& v3
) {	// L2
  int32_t v4 = v0.read();	// L3
  int32_t _st__pid0;	// L4
  _st__pid0 = v4;	// L5
  int32_t v6 = _st__pid0;	// L6
  int32_t slot;	// L7
  slot = v6;	// L8
  l_S_k_0_k: for (int k = 0; k < 8; k++) {	// L9
    int8_t v9[8];
    {
      hls::vector< int8_t, 8 > _vec = v1.read();
      for (int _iv0 = 0; _iv0 < 8; ++_iv0) {
        v9[_iv0] = _vec[_iv0];
      }
    }	// L10
    int32_t v10 = slot;	// L11
    int v11 = v10;	// L12
    int8_t v12 = v9[v11];	// L13
    v2.write(v12);	// L14
    {
      hls::vector< int8_t, 8 > _vec;
      for (int _iv0 = 0; _iv0 < 8; ++_iv0) {
        _vec[_iv0] = v9[_iv0];
      }
      v3.write(_vec);
    }	// L15
  }
}

/// This is top function.
void top(

) {	// L19
  #pragma HLS dataflow
  // Stream of vectors: each vector packs int8_t array[8] into hls::vector<int8_t, 8>
  hls::stream< hls::vector< int8_t, 8 > > v13;
  #pragma HLS stream variable=v13 depth=2	// L20
  hls::stream< int8_t > v14;
  #pragma HLS stream variable=v14 depth=2	// L21
  // Stream of vectors: each vector packs int8_t array[8] into hls::vector<int8_t, 8>
  hls::stream< hls::vector< int8_t, 8 > > v15;
  #pragma HLS stream variable=v15 depth=2	// L22
  hls::stream< int32_t > v16;
  #pragma HLS stream variable=v16 depth=1	// L23
  feed_r0_0(v16, v15, v14, v13);	// L24
}

