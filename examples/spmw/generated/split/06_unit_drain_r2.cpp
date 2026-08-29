
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
void drain_r2_0(
  hls::stream< int32_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int32_t >& v4
) {	// L2
  int32_t v5 = v0.read();	// L5
  int32_t _st__pid0;	// L6
  _st__pid0 = v5;	// L7
  int32_t v7 = v1.read();	// L8
  int32_t _st__pid1;	// L9
  _st__pid1 = v7;	// L10
  int32_t v9 = _st__pid0;	// L11
  int32_t row;	// L12
  row = v9;	// L13
  int32_t v11 = _st__pid1;	// L14
  int32_t _col;	// L15
  _col = v11;	// L16
  int32_t v13 = v2.read();	// L17
  v3.write(v13);	// L18
  int32_t v14 = row;	// L19
  int v15 = v14;	// L20
  for (int v16 = 0; v16 < v15; v16 += 1) {	// L21
    int32_t v17 = v4.read();	// L22
    v3.write(v17);	// L23
  }
}

/// This is top function.
void top(

) {	// L27
  #pragma HLS dataflow
  hls::stream< int32_t > v18;
  #pragma HLS stream variable=v18 depth=2	// L28
  hls::stream< int32_t > v19;
  #pragma HLS stream variable=v19 depth=2	// L29
  hls::stream< int32_t > v20;
  #pragma HLS stream variable=v20 depth=2	// L30
  hls::stream< int32_t > v21;
  #pragma HLS stream variable=v21 depth=1	// L31
  hls::stream< int32_t > v22;
  #pragma HLS stream variable=v22 depth=1	// L32
  drain_r2_0(v21, v22, v19, v18, v20);	// L33
}

