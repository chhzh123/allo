
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
using namespace std;
/// This is top function.
void pe_3_r2_0(
  hls::stream< float >& v0,
  hls::stream< float >& v1,
  hls::stream< float >& v2
) {	// L2
  float acc;	// L6
  acc = (float)0.000000;	// L7
  l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L8
    float v5 = v2.read();	// L9
    float a;	// L10
    a = v5;	// L11
    float v7 = v1.read();	// L12
    float b;	// L13
    b = v7;	// L14
    float v9 = a;	// L15
    float v10 = b;	// L16
    float v11 = v9 * v10;	// L17
    float v12 = acc;	// L18
    float v13 = v12 + v11;	// L19
    acc = v13;	// L20
  }
  float v14 = acc;	// L22
  v0.write(v14);	// L23
}

