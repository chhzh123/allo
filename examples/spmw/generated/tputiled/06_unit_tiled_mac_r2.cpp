
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void tiled_mac_r2_0(
  hls::stream< hls::vector< int8_t, 2 > >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int32_t >& v3
) {	// L2
  int8_t v4[2];
  {
    hls::vector< int8_t, 2 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v4[_iv0] = _vec[_iv0];
    }
  }	// L3
  l_S_m_0_m: for (int m = 0; m < 6; m++) {	// L4
    l_S_t_0_t: for (int t = 0; t < 2; t++) {	// L5
      int8_t v7 = v1.read();	// L6
      int8_t a;	// L7
      a = v7;	// L8
      int32_t p;	// L11
      p = 0;	// L12
      int8_t v10 = v4[t];	// L13
      int32_t v11 = v10;	// L14
      int32_t wt;	// L15
      wt = v11;	// L16
      int32_t v13 = p;	// L17
      int8_t v14 = a;	// L18
      int32_t v15 = wt;	// L19
      ap_int<40> v16 = v14;	// L20
      ap_int<40> v17 = v15;	// L21
      ap_int<40> v18 = v16 * v17;	// L22
      ap_int<41> v19 = v13;	// L23
      ap_int<41> v20 = v18;	// L24
      ap_int<41> v21 = v19 + v20;	// L25
      v3.write(v21);	// L26
      int8_t v22 = a;	// L27
      v2.write(v22);	// L28
    }
  }
}

