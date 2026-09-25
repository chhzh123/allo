
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
  int8_t cur;	// L7
  cur = 0;	// L8
  l_S_s_0_s: for (int s = 0; s < 262208; s++) {	// L9
  #pragma HLS pipeline II=1
    ap_int<33> v8 = s;	// L10
    bool v9 = v8 >= 64;	// L13
    if (v9) {	// L14
      int8_t v10 = v0.read();	// L15
      int8_t a;	// L16
      a = v10;	// L17
      int32_t p;	// L18
      p = 0;	// L19
      int8_t v13 = a;	// L20
      v1.write(v13);	// L21
      int32_t v14 = p;	// L22
      int8_t v15 = a;	// L23
      int8_t v16 = cur;	// L24
      int16_t v17 = v15;	// L25
      int16_t v18 = v16;	// L26
      int16_t v19 = v17 * v18;	// L27
      #pragma HLS bind_op variable=v19 op=mul impl=fabric
      ap_int<33> v20 = v14;	// L28
      ap_int<33> v21 = v19;	// L29
      ap_int<33> v22 = v20 + v21;	// L30
      v2.write(v22);	// L31
    }
    int8_t v23 = nxt;	// L33
    v4.write(v23);	// L34
    int8_t v24 = v3.read();	// L35
    nxt = v24;	// L36
    int32_t v25 = s;	// L37
    int32_t v26 = v25 & 63;	// L39
    bool v27 = v26 == 63;	// L40
    if (v27) {	// L41
      int8_t v28 = nxt;	// L42
      cur = v28;	// L43
    }
  }
}

