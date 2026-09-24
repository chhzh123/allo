
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void vpu_r0_0(
  hls::stream< hls::vector< int32_t, 2 > >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2
) {	// L2
  int32_t v3[2];
  {
    hls::vector< int32_t, 2 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v3[_iv0] = _vec[_iv0];
    }
  }	// L3
  int32_t v4 = v3[1];	// L4
  int32_t v5 = v4 & 31;	// L6
  int32_t sh;	// L7
  sh = v5;	// L8
  l_S__m_0__m: for (int _m = 0; _m < 256; _m++) {	// L9
  #pragma HLS pipeline II=1
    int32_t v8 = v2.read();	// L10
    int32_t acc;	// L11
    acc = v8;	// L12
    int32_t v10 = acc;	// L13
    bool v11 = v10 < 0;	// L15
    if (v11) {	// L16
      acc = 0;	// L17
    }
    int32_t v12 = acc;	// L19
    int32_t v13 = sh;	// L20
    int32_t v14 = v12 >> v13;	// L21
    acc = v14;	// L22
    int32_t v15 = acc;	// L23
    bool v16 = v15 > 127;	// L25
    if (v16) {	// L26
      acc = 127;	// L27
    }
    int32_t v17 = acc;	// L29
    v1.write(v17);	// L30
  }
}

