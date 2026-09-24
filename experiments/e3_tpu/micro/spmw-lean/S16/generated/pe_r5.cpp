
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
void pe_r5_0(
  hls::stream< int8_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int8_t >& v3
) {	// L2
  int8_t nxt;	// L5
  nxt = 0;	// L6
  l_S__k_0__k: for (int _k = 0; _k < 16; _k++) {	// L7
  #pragma HLS pipeline II=1
    int8_t v6 = v3.read();	// L8
    nxt = v6;	// L9
  }
  int8_t v7 = nxt;	// L11
  int8_t cur;	// L12
  cur = v7;	// L13
  l_S_s_1_s: for (int s = 0; s < 256; s++) {	// L14
  #pragma HLS pipeline II=1
    int8_t v10 = v0.read();	// L15
    int8_t a;	// L16
    a = v10;	// L17
    int32_t v12 = v1.read();	// L18
    int32_t p;	// L19
    p = v12;	// L20
    int32_t v14 = p;	// L21
    int8_t v15 = a;	// L22
    int8_t v16 = cur;	// L23
    int16_t v17 = v15;	// L24
    int16_t v18 = v16;	// L25
    int16_t v19 = v17 * v18;	// L26
    #pragma HLS bind_op variable=v19 op=mul impl=fabric
    ap_int<33> v20 = v14;	// L27
    ap_int<33> v21 = v19;	// L28
    ap_int<33> v22 = v20 + v21;	// L29
    v2.write(v22);	// L30
    int8_t v23 = v3.read();	// L31
    nxt = v23;	// L32
    int32_t v24 = s;	// L33
    int32_t v25 = v24 & 15;	// L35
    bool v26 = v25 == 15;	// L36
    if (v26) {	// L37
      int8_t v27 = nxt;	// L38
      cur = v27;	// L39
    }
  }
}

