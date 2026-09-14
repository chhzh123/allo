
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void pe_c_out_drain_io_0(
  int32_t v0[4][4],
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2
) {	// L2
  int32_t v3 = v2.read();	// L3
  int32_t _q0;	// L4
  _q0 = v3;	// L5
  l_S__t_0__t: for (int _t = 0; _t < 4; _t++) {	// L6
    int32_t v6 = v1.read();	// L7
    int32_t v7 = _q0;	// L8
    int v8 = v7;	// L9
    v0[v8][_t] = v6;	// L10
  }
}

