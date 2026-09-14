
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void mac_r2_0(
  hls::stream< int8_t >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int32_t >& v3
) {	// L2
  int8_t v4 = v0.read();	// L3
  int8_t _st_w;	// L4
  _st_w = v4;	// L5
  l_S_m_0_m: for (int m = 0; m < 6; m++) {	// L6
    int8_t v7 = v1.read();	// L7
    int8_t a;	// L8
    a = v7;	// L9
    int32_t p;	// L12
    p = 0;	// L13
    int32_t v10 = p;	// L14
    int8_t v11 = a;	// L15
    int8_t v12 = _st_w;	// L16
    int16_t v13 = v11;	// L17
    int16_t v14 = v12;	// L18
    int16_t v15 = v13 * v14;	// L19
    ap_int<33> v16 = v10;	// L20
    ap_int<33> v17 = v15;	// L21
    ap_int<33> v18 = v16 + v17;	// L22
    v3.write(v18);	// L23
    int8_t v19 = a;	// L24
    v2.write(v19);	// L25
  }
}

