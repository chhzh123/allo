
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
using namespace std;
void pe_1_r2_0(
  hls::stream< float >& v0,
  hls::stream< float >& v1,
  hls::stream< float >& v2
) {	// L2
  float acc;	// L4
  acc = (float)0.000000;	// L5
  l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L6
    float v5 = v0.read();	// L7
    float a;	// L8
    a = v5;	// L9
    float v7 = v1.read();	// L10
    float b;	// L11
    b = v7;	// L12
    float v9 = a;	// L13
    float v10 = b;	// L14
    float v11 = v9 * v10;	// L15
    float v12 = acc;	// L16
    float v13 = v12 + v11;	// L17
    acc = v13;	// L18
  }
  float v14 = acc;	// L20
  v2.write(v14);	// L21
}

/// This is top function.
void top(

) {	// L24
  #pragma HLS dataflow
  hls::stream< float > v15;
  #pragma HLS stream variable=v15 depth=2	// L25
  hls::stream< float > v16;
  #pragma HLS stream variable=v16 depth=2	// L26
  hls::stream< float > v17;
  #pragma HLS stream variable=v17 depth=2	// L27
  pe_1_r2_0(v17, v16, v15);	// L28
}

