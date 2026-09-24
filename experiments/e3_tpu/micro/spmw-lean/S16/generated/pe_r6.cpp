
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
void pe_r6_0(
  hls::stream< int8_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int8_t >& v2
) {	// L2
  int8_t nxt;	// L5
  nxt = 0;	// L6
  l_S__k_0__k: for (int _k = 0; _k < 16; _k++) {	// L7
  #pragma HLS pipeline II=1
    int8_t v5 = v2.read();	// L8
    nxt = v5;	// L9
  }
  int8_t v6 = nxt;	// L11
  int8_t cur;	// L12
  cur = v6;	// L13
  l_S_s_1_s: for (int s = 0; s < 256; s++) {	// L14
  #pragma HLS pipeline II=1
    int8_t v9 = v0.read();	// L15
    int8_t a;	// L16
    a = v9;	// L17
    int32_t p;	// L18
    p = 0;	// L19
    int32_t v12 = p;	// L20
    int8_t v13 = a;	// L21
    int8_t v14 = cur;	// L22
    int16_t v15 = v13;	// L23
    int16_t v16 = v14;	// L24
    int16_t v17 = v15 * v16;	// L25
    #pragma HLS bind_op variable=v17 op=mul impl=fabric
    ap_int<33> v18 = v12;	// L26
    ap_int<33> v19 = v17;	// L27
    ap_int<33> v20 = v18 + v19;	// L28
    v1.write(v20);	// L29
    int8_t v21 = v2.read();	// L30
    nxt = v21;	// L31
    int32_t v22 = s;	// L32
    int32_t v23 = v22 & 15;	// L34
    bool v24 = v23 == 15;	// L35
    if (v24) {	// L36
      int8_t v25 = nxt;	// L37
      cur = v25;	// L38
    }
  }
}

