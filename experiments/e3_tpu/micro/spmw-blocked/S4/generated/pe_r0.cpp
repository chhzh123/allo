
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
void pe_r0_0(
  hls::stream< int8_t >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int8_t >& v4,
  hls::stream< int8_t >& v5
) {	// L2
  int8_t nxt;	// L5
  nxt = 0;	// L6
  int8_t cur;	// L7
  cur = 0;	// L8
  l_S_s_0_s: for (int s = 0; s < 4112; s++) {	// L9
  #pragma HLS pipeline II=1
    ap_int<33> v9 = s;	// L10
    bool v10 = v9 >= 16;	// L13
    if (v10) {	// L14
      int8_t v11 = v0.read();	// L15
      int8_t a;	// L16
      a = v11;	// L17
      int32_t v13 = v2.read();	// L18
      int32_t p;	// L19
      p = v13;	// L20
      int8_t v15 = a;	// L21
      v1.write(v15);	// L22
      int32_t v16 = p;	// L23
      int8_t v17 = a;	// L24
      int8_t v18 = cur;	// L25
      int16_t v19 = v17;	// L26
      int16_t v20 = v18;	// L27
      int16_t v21 = v19 * v20;	// L28
      #pragma HLS bind_op variable=v21 op=mul impl=fabric
      ap_int<33> v22 = v16;	// L29
      ap_int<33> v23 = v21;	// L30
      ap_int<33> v24 = v22 + v23;	// L31
      v3.write(v24);	// L32
    }
    int8_t v25 = nxt;	// L34
    v5.write(v25);	// L35
    int8_t v26 = v4.read();	// L36
    nxt = v26;	// L37
    int32_t v27 = s;	// L38
    int32_t v28 = v27 & 15;	// L40
    bool v29 = v28 == 15;	// L41
    if (v29) {	// L42
      int8_t v30 = nxt;	// L43
      cur = v30;	// L44
    }
  }
}

