
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
void pe_r7_0(
  hls::stream< int8_t >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int8_t >& v4,
  hls::stream< int8_t >& v5
) {	// L2
  int8_t nxt;	// L5
  nxt = 0;	// L6
  l_S__k_0__k: for (int _k = 0; _k < 16; _k++) {	// L7
  #pragma HLS pipeline II=1
    int8_t v8 = nxt;	// L8
    v5.write(v8);	// L9
    int8_t v9 = v4.read();	// L10
    nxt = v9;	// L11
  }
  int8_t v10 = nxt;	// L13
  int8_t cur;	// L14
  cur = v10;	// L15
  l_S_s_1_s: for (int s = 0; s < 256; s++) {	// L16
  #pragma HLS pipeline II=1
    int8_t v13 = v0.read();	// L17
    int8_t a;	// L18
    a = v13;	// L19
    int32_t v15 = v2.read();	// L20
    int32_t p;	// L21
    p = v15;	// L22
    int8_t v17 = a;	// L23
    v1.write(v17);	// L24
    int32_t v18 = p;	// L25
    int8_t v19 = a;	// L26
    int8_t v20 = cur;	// L27
    int16_t v21 = v19;	// L28
    int16_t v22 = v20;	// L29
    int16_t v23 = v21 * v22;	// L30
    #pragma HLS bind_op variable=v23 op=mul impl=fabric
    ap_int<33> v24 = v18;	// L31
    ap_int<33> v25 = v23;	// L32
    ap_int<33> v26 = v24 + v25;	// L33
    v3.write(v26);	// L34
    int8_t v27 = nxt;	// L35
    v5.write(v27);	// L36
    int8_t v28 = v4.read();	// L37
    nxt = v28;	// L38
    int32_t v29 = s;	// L39
    int32_t v30 = v29 & 15;	// L41
    bool v31 = v30 == 15;	// L42
    if (v31) {	// L43
      int8_t v32 = nxt;	// L44
      cur = v32;	// L45
    }
  }
}

