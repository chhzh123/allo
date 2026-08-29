
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
  hls::stream< int8_t >& v3,
  hls::stream< int32_t >& v4
) {	// L2
  int32_t acc;	// L4
  acc = 0;	// L5
  l_S_k_0_k: for (int k = 0; k < 3; k++) {	// L6
    int8_t v7 = v0.read();	// L7
    int8_t a;	// L8
    a = v7;	// L9
    int8_t v9 = v1.read();	// L10
    int8_t b;	// L11
    b = v9;	// L12
    int8_t v11 = a;	// L13
    int8_t v12 = b;	// L14
    int16_t v13 = v11;	// L15
    int16_t v14 = v12;	// L16
    int16_t v15 = v13 * v14;	// L17
    int32_t v16 = acc;	// L18
    ap_int<33> v17 = v16;	// L19
    ap_int<33> v18 = v15;	// L20
    ap_int<33> v19 = v17 + v18;	// L21
    int32_t v20 = v19;	// L22
    acc = v20;	// L23
    int8_t v21 = a;	// L24
    v2.write(v21);	// L25
    int8_t v22 = b;	// L26
    v3.write(v22);	// L27
  }
  int32_t v23 = acc;	// L29
  v4.write(v23);	// L30
}

/// This is top function.
void top(

) {	// L33
  #pragma HLS dataflow
  hls::stream< int32_t > v24;
  #pragma HLS stream variable=v24 depth=2	// L34
  hls::stream< int8_t > v25;
  #pragma HLS stream variable=v25 depth=2	// L35
  hls::stream< int8_t > v26;
  #pragma HLS stream variable=v26 depth=2	// L36
  hls::stream< int8_t > v27;
  #pragma HLS stream variable=v27 depth=2	// L37
  hls::stream< int8_t > v28;
  #pragma HLS stream variable=v28 depth=2	// L38
  pe_r2_0(v28, v26, v25, v27, v24);	// L39
}

