
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
using namespace std;
void pe_r2_0(
  hls::stream< float >& v0,
  hls::stream< float >& v1,
  hls::stream< float >& v2,
  hls::stream< float >& v3,
  hls::stream< float >& v4
) {	// L2
  float acc;	// L4
  acc = (float)0.000000;	// L5
  l_S_k_0_k: for (int k = 0; k < 3; k++) {	// L6
    float v7 = v0.read();	// L7
    float a;	// L8
    a = v7;	// L9
    float v9 = v1.read();	// L10
    float b;	// L11
    b = v9;	// L12
    float v11 = a;	// L13
    float v12 = b;	// L14
    float v13 = v11 * v12;	// L15
    float v14 = acc;	// L16
    float v15 = v14 + v13;	// L17
    acc = v15;	// L18
    float v16 = a;	// L19
    v2.write(v16);	// L20
    float v17 = b;	// L21
    v3.write(v17);	// L22
  }
  float v18 = acc;	// L24
  v4.write(v18);	// L25
}

/// This is top function.
void top(

) {	// L28
  #pragma HLS dataflow
  hls::stream< float > v19;
  #pragma HLS stream variable=v19 depth=2	// L29
  hls::stream< float > v20;
  #pragma HLS stream variable=v20 depth=2	// L30
  hls::stream< float > v21;
  #pragma HLS stream variable=v21 depth=2	// L31
  hls::stream< float > v22;
  #pragma HLS stream variable=v22 depth=2	// L32
  hls::stream< float > v23;
  #pragma HLS stream variable=v23 depth=2	// L33
  pe_r2_0(v23, v21, v20, v22, v19);	// L34
}

