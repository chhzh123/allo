
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
void pe_r3_0(
  hls::stream< int8_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int8_t >& v3
) {	// L2
  int8_t nxt;	// L5
  nxt = 0;	// L6
  int8_t cur;	// L7
  cur = 0;	// L8
  l_S_s_0_s: for (int s = 0; s < 65600; s++) {	// L9
  #pragma HLS pipeline II=1
    ap_int<33> v7 = s;	// L10
    bool v8 = v7 >= 64;	// L13
    if (v8) {	// L14
      int8_t v9 = v0.read();	// L15
      int8_t a;	// L16
      a = v9;	// L17
      int32_t v11 = v1.read();	// L18
      int32_t p;	// L19
      p = v11;	// L20
      int32_t v13 = p;	// L21
      int8_t v14 = a;	// L22
      int8_t v15 = cur;	// L23
      int16_t v16 = v14;	// L24
      int16_t v17 = v15;	// L25
      int16_t v18 = v16 * v17;	// L26
      #pragma HLS bind_op variable=v18 op=mul impl=fabric
      ap_int<33> v19 = v13;	// L27
      ap_int<33> v20 = v18;	// L28
      ap_int<33> v21 = v19 + v20;	// L29
      v2.write(v21);	// L30
    }
    int8_t v22 = v3.read();	// L32
    nxt = v22;	// L33
    int32_t v23 = s;	// L34
    int32_t v24 = v23 & 63;	// L36
    bool v25 = v24 == 63;	// L37
    if (v25) {	// L38
      int8_t v26 = nxt;	// L39
      cur = v26;	// L40
    }
  }
}

