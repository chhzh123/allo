
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
void act_r0_0(
  hls::stream< int32_t >& v0,
  hls::stream< int8_t >& v1
) {	// L2
  l_S_m_0_m: for (int m = 0; m < 6; m++) {	// L5
    int32_t v3 = v0.read();	// L6
    int32_t z;	// L7
    z = v3;	// L8
    int32_t v5 = z;	// L9
    bool v6 = v5 < 0;	// L10
    if (v6) {	// L11
      z = 0;	// L12
    }
    int32_t v7 = z;	// L14
    int32_t v8 = v7 >> 2;	// L15
    int8_t v9 = v8;	// L16
    int8_t y;	// L17
    y = v9;	// L18
    int8_t v11 = y;	// L19
    v1.write(v11);	// L20
  }
}

/// This is top function.
void top(

) {	// L24
  #pragma HLS dataflow
  hls::stream< int8_t > v12;
  #pragma HLS stream variable=v12 depth=2	// L25
  hls::stream< int32_t > v13;
  #pragma HLS stream variable=v13 depth=2	// L26
  act_r0_0(v13, v12);	// L27
}

