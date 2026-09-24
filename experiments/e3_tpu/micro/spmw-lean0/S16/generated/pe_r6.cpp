
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
  int8_t cur;	// L7
  cur = 0;	// L8
  l_S_s_0_s: for (int s = 0; s < 272; s++) {	// L9
  #pragma HLS pipeline II=1
    ap_int<33> v6 = s;	// L10
    bool v7 = v6 >= 16;	// L13
    if (v7) {	// L14
      int8_t v8 = v0.read();	// L15
      int8_t a;	// L16
      a = v8;	// L17
      int32_t p;	// L18
      p = 0;	// L19
      int32_t v11 = p;	// L20
      int8_t v12 = a;	// L21
      int8_t v13 = cur;	// L22
      int16_t v14 = v12;	// L23
      int16_t v15 = v13;	// L24
      int16_t v16 = v14 * v15;	// L25
      #pragma HLS bind_op variable=v16 op=mul impl=fabric
      ap_int<33> v17 = v11;	// L26
      ap_int<33> v18 = v16;	// L27
      ap_int<33> v19 = v17 + v18;	// L28
      v1.write(v19);	// L29
    }
    int8_t v20 = v2.read();	// L31
    nxt = v20;	// L32
    int32_t v21 = s;	// L33
    int32_t v22 = v21 & 15;	// L35
    bool v23 = v22 == 15;	// L36
    if (v23) {	// L37
      int8_t v24 = nxt;	// L38
      cur = v24;	// L39
    }
  }
}

