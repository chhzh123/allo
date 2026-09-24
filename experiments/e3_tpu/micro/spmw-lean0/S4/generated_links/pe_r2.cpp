
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
  hls::stream< int8_t >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int8_t >& v3,
  hls::stream< int8_t >& v4
) {	// L2
  int8_t nxt;	// L5
  nxt = 0;	// L6
  l_S__k_0__k: for (int _k = 0; _k < 4; _k++) {	// L7
  #pragma HLS pipeline II=1
    int8_t v7 = nxt;	// L8
    v4.write(v7);	// L9
    int8_t v8 = v3.read();	// L10
    nxt = v8;	// L11
  }
  int8_t v9 = nxt;	// L13
  int8_t cur;	// L14
  cur = v9;	// L15
  l_S_s_1_s: for (int s = 0; s < 64; s++) {	// L16
  #pragma HLS pipeline II=1
    int8_t v12 = v0.read();	// L17
    int8_t a;	// L18
    a = v12;	// L19
    int32_t p;	// L20
    p = 0;	// L21
    int8_t v15 = a;	// L22
    v1.write(v15);	// L23
    int32_t v16 = p;	// L24
    int8_t v17 = a;	// L25
    int8_t v18 = cur;	// L26
    int16_t v19 = v17;	// L27
    int16_t v20 = v18;	// L28
    int16_t v21 = v19 * v20;	// L29
    #pragma HLS bind_op variable=v21 op=mul impl=fabric
    ap_int<33> v22 = v16;	// L30
    ap_int<33> v23 = v21;	// L31
    ap_int<33> v24 = v22 + v23;	// L32
    v2.write(v24);	// L33
    int8_t v25 = nxt;	// L34
    v4.write(v25);	// L35
    int8_t v26 = v3.read();	// L36
    nxt = v26;	// L37
    int32_t v27 = s;	// L38
    int32_t v28 = v27 & 3;	// L40
    bool v29 = v28 == 3;	// L41
    if (v29) {	// L42
      int8_t v30 = nxt;	// L43
      cur = v30;	// L44
    }
  }
}

