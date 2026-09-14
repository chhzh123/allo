
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
using namespace std;
/// This is top function.
void pe_r2_0(
  hls::stream< float >& v0,
  hls::stream< float >& v1,
  hls::stream< float >& v2,
  hls::stream< float >& v3,
  hls::stream< float >& v4
) {	// L2
  float acc;	// L6
  acc = (float)0.000000;	// L7
  l_S_k_0_k: for (int k = 0; k < 3; k++) {	// L8
    float v7 = v4.read();	// L9
    float a;	// L10
    a = v7;	// L11
    float v9 = v2.read();	// L12
    float b;	// L13
    b = v9;	// L14
    float v11 = a;	// L15
    float v12 = b;	// L16
    float v13 = v11 * v12;	// L17
    float v14 = acc;	// L18
    float v15 = v14 + v13;	// L19
    acc = v15;	// L20
    float v16 = a;	// L21
    v1.write(v16);	// L22
    float v17 = b;	// L23
    v3.write(v17);	// L24
  }
  float v18 = acc;	// L26
  v0.write(v18);	// L27
}

