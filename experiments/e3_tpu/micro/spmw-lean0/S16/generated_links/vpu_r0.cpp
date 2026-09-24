
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
  int32_t v4 = v3[0];	// L4
  int32_t bias;	// L5
  bias = v4;	// L6
  int32_t v6 = v3[1];	// L7
  int32_t sh;	// L8
  sh = v6;	// L9
  l_S__m_0__m: for (int _m = 0; _m < 256; _m++) {	// L10
    int32_t v9 = v2.read();	// L11
    int32_t z;	// L12
    z = v9;	// L13
    int32_t v11 = bias;	// L14
    int32_t v12 = z;	// L15
    ap_int<33> v13 = v11;	// L16
    ap_int<33> v14 = v12;	// L17
    ap_int<33> v15 = v13 + v14;	// L18
    int32_t v16 = v15;	// L19
    int32_t acc;	// L20
    acc = v16;	// L21
    int32_t v18 = acc;	// L22
    bool v19 = v18 < 0;	// L25
    if (v19) {	// L26
      acc = 0;	// L29
    }
    int32_t v20 = acc;	// L31
    int32_t v21 = sh;	// L32
    int32_t v22 = v20 >> v21;	// L33
    acc = v22;	// L34
    int32_t v23 = acc;	// L35
    bool v24 = v23 > 127;	// L38
    if (v24) {	// L39
      acc = 127;	// L42
    }
    int32_t v25 = acc;	// L44
    v1.write(v25);	// L45
  }
}

