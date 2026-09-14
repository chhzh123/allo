
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
  hls::stream< int8_t >& v3
) {	// L2
  int32_t acc;	// L5
  acc = 0;	// L6
  l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L7
    int8_t v6 = v3.read();	// L8
    int8_t a;	// L9
    a = v6;	// L10
    int8_t v8 = v2.read();	// L11
    int8_t b;	// L12
    b = v8;	// L13
    int8_t v10 = a;	// L14
    int8_t v11 = b;	// L15
    int16_t v12 = v10;	// L16
    int16_t v13 = v11;	// L17
    int16_t v14 = v12 * v13;	// L18
    int32_t v15 = acc;	// L19
    ap_int<33> v16 = v15;	// L20
    ap_int<33> v17 = v14;	// L21
    ap_int<33> v18 = v16 + v17;	// L22
    int32_t v19 = v18;	// L23
    acc = v19;	// L24
    int8_t v20 = a;	// L25
    v1.write(v20);	// L26
  }
  int32_t v21 = acc;	// L28
  v0.write(v21);	// L29
}

