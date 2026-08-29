
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
void pe_c_out_drain_io_0(
  int32_t v0[4][4],
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2
) {	// L2
  int32_t v3 = v1.read();	// L3
  int32_t _st__pid0;	// L4
  _st__pid0 = v3;	// L5
  int32_t v5 = _st__pid0;	// L6
  int32_t _q0;	// L7
  _q0 = v5;	// L8
  l_S__t_0__t: for (int _t = 0; _t < 4; _t++) {	// L9
    int32_t v8 = v2.read();	// L10
    int32_t v9 = _q0;	// L11
    int v10 = v9;	// L12
    v0[v10][_t] = v8;	// L13
  }
}

/// This is top function.
void top(
  int32_t v11[4][4]
) {	// L17
  #pragma HLS dataflow
  hls::stream< int32_t > v12;
  #pragma HLS stream variable=v12 depth=2	// L18
  hls::stream< int32_t > v13;
  #pragma HLS stream variable=v13 depth=1	// L19
  pe_c_out_drain_io_0(v11, v13, v12);	// L20
}

