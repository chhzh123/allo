
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
  hls::stream< int32_t >& v3
) {	// L2
  int8_t v4 = v0.read();	// L3
  int8_t _st_w;	// L4
  _st_w = v4;	// L5
  l_S_m_0_m: for (int m = 0; m < 6; m++) {	// L6
    int8_t v7 = v1.read();	// L7
    int8_t a;	// L8
    a = v7;	// L9
    int32_t v9 = v2.read();	// L10
    int32_t p;	// L11
    p = v9;	// L12
    int32_t v11 = p;	// L13
    int8_t v12 = a;	// L14
    int8_t v13 = _st_w;	// L15
    int16_t v14 = v12;	// L16
    int16_t v15 = v13;	// L17
    int16_t v16 = v14 * v15;	// L18
    ap_int<33> v17 = v11;	// L19
    ap_int<33> v18 = v16;	// L20
    ap_int<33> v19 = v17 + v18;	// L21
    v3.write(v19);	// L22
  }
}

/// This is top function.
void top(

) {	// L26
  #pragma HLS dataflow
  hls::stream< int8_t > v20;
  #pragma HLS stream variable=v20 depth=2	// L27
  hls::stream< int8_t > v21;
  #pragma HLS stream variable=v21 depth=2	// L28
  hls::stream< int32_t > v22;
  #pragma HLS stream variable=v22 depth=2	// L29
  hls::stream< int32_t > v23;
  #pragma HLS stream variable=v23 depth=2	// L30
  mac_r2_0(v20, v21, v22, v23);	// L31
}

