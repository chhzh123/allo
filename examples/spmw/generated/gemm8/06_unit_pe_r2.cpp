
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
void pe_r2_0(
  hls::stream< int32_t >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int8_t >& v3,
  hls::stream< int8_t >& v4
) {	// L2
  int32_t acc;	// L5
  acc = 0;	// L6
  l_S_k_0_k: for (int k = 0; k < 3; k++) {	// L7
    int8_t v7 = v4.read();	// L8
    int8_t a;	// L9
    a = v7;	// L10
    int8_t v9 = v2.read();	// L11
    int8_t b;	// L12
    b = v9;	// L13
    int8_t v11 = a;	// L14
    int8_t v12 = b;	// L15
    int16_t v13 = v11;	// L16
    int16_t v14 = v12;	// L17
    int16_t v15 = v13 * v14;	// L18
    int32_t v16 = acc;	// L19
    ap_int<33> v17 = v16;	// L20
    ap_int<33> v18 = v15;	// L21
    ap_int<33> v19 = v17 + v18;	// L22
    int32_t v20 = v19;	// L23
    acc = v20;	// L24
    int8_t v21 = a;	// L25
    v1.write(v21);	// L26
    int8_t v22 = b;	// L27
    v3.write(v22);	// L28
  }
  int32_t v23 = acc;	// L30
  v0.write(v23);	// L31
}

