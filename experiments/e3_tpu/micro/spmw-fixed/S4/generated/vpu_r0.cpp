
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
  }	// L5
  int32_t v4 = v3[0];	// L6
  int32_t bias;	// L7
  bias = v4;	// L8
  int32_t v6 = v3[1];	// L9
  int32_t sh;	// L10
  sh = v6;	// L11
  l_S__m_0__m: for (int _m = 0; _m < 64; _m++) {	// L12
    int32_t v9 = v1.read();	// L13
    int32_t z;	// L14
    z = v9;	// L15
    int32_t v11 = bias;	// L16
    int32_t v12 = z;	// L17
    ap_int<33> v13 = v11;	// L18
    ap_int<33> v14 = v12;	// L19
    ap_int<33> v15 = v13 + v14;	// L20
    int32_t v16 = v15;	// L21
    int32_t acc;	// L22
    acc = v16;	// L23
    int32_t v18 = acc;	// L24
    bool v19 = v18 < 0;	// L25
    if (v19) {	// L26
      acc = 0;	// L27
    }
    int32_t v20 = acc;	// L29
    int32_t v21 = sh;	// L30
    int32_t v22 = v20 >> v21;	// L31
    acc = v22;	// L32
    int32_t v23 = acc;	// L33
    bool v24 = v23 > 127;	// L34
    if (v24) {	// L35
      acc = 127;	// L36
    }
    int32_t v25 = acc;	// L38
    v2.write(v25);	// L39
  }
}

/// This is top function.
void top(

) {	// L43
  #pragma HLS dataflow
  // Stream of vectors: each vector packs int32_t array[2] into hls::vector<int32_t, 2>
  hls::stream< hls::vector< int32_t, 2 > > v26;
  #pragma HLS stream variable=v26 depth=2	// L44
  hls::stream< int32_t > v27;
  #pragma HLS stream variable=v27 depth=2	// L45
  hls::stream< int32_t > v28;
  #pragma HLS stream variable=v28 depth=4	// L46
  vpu_r0_0(v26, v28, v27);	// L47
}

