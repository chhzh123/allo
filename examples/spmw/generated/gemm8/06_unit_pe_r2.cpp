
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
void pe_r2_0(
  hls::stream< int8_t >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int32_t >& v3
) {	// L2
  int32_t acc;	// L4
  acc = 0;	// L5
  l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L6
    int8_t v6 = v0.read();	// L7
    int8_t a;	// L8
    a = v6;	// L9
    int8_t v8 = v1.read();	// L10
    int8_t b;	// L11
    b = v8;	// L12
    int8_t v10 = a;	// L13
    int8_t v11 = b;	// L14
    int16_t v12 = v10;	// L15
    int16_t v13 = v11;	// L16
    int16_t v14 = v12 * v13;	// L17
    int32_t v15 = acc;	// L18
    ap_int<33> v16 = v15;	// L19
    ap_int<33> v17 = v14;	// L20
    ap_int<33> v18 = v16 + v17;	// L21
    int32_t v19 = v18;	// L22
    acc = v19;	// L23
    int8_t v20 = a;	// L24
    v2.write(v20);	// L25
  }
  int32_t v21 = acc;	// L27
  v3.write(v21);	// L28
}

/// This is top function.
void top(

) {	// L31
  #pragma HLS dataflow
  hls::stream< int32_t > v22;
  #pragma HLS stream variable=v22 depth=2	// L32
  hls::stream< int8_t > v23;
  #pragma HLS stream variable=v23 depth=2	// L33
  hls::stream< int8_t > v24;
  #pragma HLS stream variable=v24 depth=2	// L34
  hls::stream< int8_t > v25;
  #pragma HLS stream variable=v25 depth=2	// L35
  pe_r2_0(v25, v24, v23, v22);	// L36
}

