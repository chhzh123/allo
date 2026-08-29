
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>
using namespace std;
void feed_2_r2_0(
  hls::stream< int32_t >& v0,
  hls::stream< hls::vector< int8_t, 4 > >& v1,
  hls::stream< int8_t >& v2
) {	// L2
  int32_t v3 = v0.read();	// L3
  int32_t _st__pid0;	// L4
  _st__pid0 = v3;	// L5
  int32_t v5 = _st__pid0;	// L6
  int32_t slot;	// L7
  slot = v5;	// L8
  l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L9
    int8_t v8[4];
    {
      hls::vector< int8_t, 4 > _vec = v1.read();
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        v8[_iv0] = _vec[_iv0];
      }
    }	// L10
    int32_t v9 = slot;	// L11
    int v10 = v9;	// L12
    int8_t v11 = v8[v10];	// L13
    v2.write(v11);	// L14
  }
}

/// This is top function.
void top(

) {	// L18
  #pragma HLS dataflow
  hls::stream< int8_t > v12;
  #pragma HLS stream variable=v12 depth=2	// L19
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v13;
  #pragma HLS stream variable=v13 depth=2	// L20
  hls::stream< int32_t > v14;
  #pragma HLS stream variable=v14 depth=1	// L21
  feed_2_r2_0(v14, v13, v12);	// L22
}

