
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void act_r0_0(
  hls::stream< int8_t >& v0,
  hls::stream< int32_t >& v1
) {	// L2
  l_S_m_0_m: for (int m = 0; m < 6; m++) {	// L3
    int32_t v3 = v1.read();	// L4
    int32_t z;	// L5
    z = v3;	// L6
    int32_t v5 = z;	// L7
    bool v6 = v5 < 0;	// L10
    if (v6) {	// L11
      z = 0;	// L14
    }
    int32_t v7 = z;	// L16
    int32_t v8 = v7 >> 2;	// L19
    int8_t v9 = v8;	// L20
    int8_t y;	// L21
    y = v9;	// L22
    int8_t v11 = y;	// L23
    v0.write(v11);	// L24
  }
}

