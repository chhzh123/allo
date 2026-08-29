
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
void mac_r2_0(
  hls::stream< int8_t >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int8_t >& v3
) {	// L2
  int8_t v4 = v0.read();	// L4
  int8_t _st_w;	// L5
  _st_w = v4;	// L6
  l_S_m_0_m: for (int m = 0; m < 6; m++) {	// L7
    int8_t v7 = v1.read();	// L8
    int8_t a;	// L9
    a = v7;	// L10
    int32_t p;	// L11
    p = 0;	// L12
    int32_t v10 = p;	// L13
    int8_t v11 = a;	// L14
    int8_t v12 = _st_w;	// L15
    int16_t v13 = v11;	// L16
    int16_t v14 = v12;	// L17
    int16_t v15 = v13 * v14;	// L18
    ap_int<33> v16 = v10;	// L19
    ap_int<33> v17 = v15;	// L20
    ap_int<33> v18 = v16 + v17;	// L21
    v2.write(v18);	// L22
    int8_t v19 = a;	// L23
    v3.write(v19);	// L24
  }
}

/// This is top function.
void top(

) {	// L28
  #pragma HLS dataflow
  hls::stream< int8_t > v20;
  #pragma HLS stream variable=v20 depth=2	// L29
  hls::stream< int8_t > v21;
  #pragma HLS stream variable=v21 depth=2	// L30
  hls::stream< int8_t > v22;
  #pragma HLS stream variable=v22 depth=2	// L31
  hls::stream< int32_t > v23;
  #pragma HLS stream variable=v23 depth=2	// L32
  mac_r2_0(v20, v21, v23, v22);	// L33
}

