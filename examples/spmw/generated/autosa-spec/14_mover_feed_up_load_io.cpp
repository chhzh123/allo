
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>
using namespace std;
void feed_up_load_io_0(
  int8_t v0[4][4],
  hls::stream< hls::vector< int8_t, 4 > >& v1
) {	// L2
  int32_t _q0;	// L5
  _q0 = 0;	// L6
  l_S__t_0__t: for (int _t = 0; _t < 4; _t++) {	// L7
    int8_t _blk[4];	// L8
    for (int v5 = 0; v5 < 4; v5++) {	// L9
      _blk[v5] = 0;	// L9
    }
    l_S__b0_0__b0: for (int _b0 = 0; _b0 < 4; _b0++) {	// L10
      int8_t v7 = v0[_t][_b0];	// L11
      _blk[_b0] = v7;	// L12
    }
    {
      hls::vector< int8_t, 4 > _vec;
      for (int _iv0 = 0; _iv0 < 4; ++_iv0) {
        _vec[_iv0] = _blk[_iv0];
      }
      v1.write(_vec);
    }	// L14
  }
}

/// This is top function.
void top(
  int8_t v8[4][4]
) {	// L18
  #pragma HLS dataflow
  // Stream of vectors: each vector packs int8_t array[4] into hls::vector<int8_t, 4>
  hls::stream< hls::vector< int8_t, 4 > > v9;
  #pragma HLS stream variable=v9 depth=2	// L19
  feed_up_load_io_0(v8, v9);	// L20
}

