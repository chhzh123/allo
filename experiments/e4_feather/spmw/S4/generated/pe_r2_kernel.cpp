
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
void pe_r2_0(
  hls::stream< hls::vector< int8_t, 2 > >& v0,
  hls::stream< hls::vector< int8_t, 8 > >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3
) {	// L2
  int8_t v4[2];
  {
    hls::vector< int8_t, 2 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v4[_iv0] = _vec[_iv0];
    }
  }	// L4
  int8_t v5[8];
  {
    hls::vector< int8_t, 8 > _vec = v1.read();
    for (int _iv0 = 0; _iv0 < 8; ++_iv0) {
      v5[_iv0] = _vec[_iv0];
    }
  }	// L5
  l_S_i_0_i: for (int i = 0; i < 4; i++) {	// L6
    int32_t p0;	// L7
    p0 = 0;	// L8
    int32_t p1;	// L9
    p1 = 0;	// L10
    int8_t v9 = v4[0];	// L11
    int32_t v10 = v9;	// L12
    int32_t x0;	// L13
    x0 = v10;	// L14
    int8_t v12 = v4[1];	// L15
    int32_t v13 = v12;	// L16
    int32_t x1;	// L17
    x1 = v13;	// L18
    int8_t v15 = v5[i];	// L19
    int32_t v16 = v15;	// L20
    int32_t w0;	// L21
    w0 = v16;	// L22
    int8_t v18 = v5[(i + 4)];	// L23
    int32_t v19 = v18;	// L24
    int32_t w1;	// L25
    w1 = v19;	// L26
    int32_t v21 = p0;	// L27
    int32_t v22 = x0;	// L28
    int32_t v23 = w0;	// L29
    int64_t v24 = v22;	// L30
    int64_t v25 = v23;	// L31
    int64_t v26 = v24 * v25;	// L32
    ap_int<65> v27 = v21;	// L33
    ap_int<65> v28 = v26;	// L34
    ap_int<65> v29 = v27 + v28;	// L35
    v2.write(v29);	// L36
    int32_t v30 = p1;	// L37
    int32_t v31 = x1;	// L38
    int32_t v32 = w1;	// L39
    int64_t v33 = v31;	// L40
    int64_t v34 = v32;	// L41
    int64_t v35 = v33 * v34;	// L42
    ap_int<65> v36 = v30;	// L43
    ap_int<65> v37 = v35;	// L44
    ap_int<65> v38 = v36 + v37;	// L45
    v3.write(v38);	// L46
  }
}

/// This is top function.
void top(

) {	// L50
  #pragma HLS dataflow
  // Stream of vectors: each vector packs int8_t array[8] into hls::vector<int8_t, 8>
  hls::stream< hls::vector< int8_t, 8 > > v39;
  #pragma HLS stream variable=v39 depth=2	// L51
  // Stream of vectors: each vector packs int8_t array[2] into hls::vector<int8_t, 2>
  hls::stream< hls::vector< int8_t, 2 > > v40;
  #pragma HLS stream variable=v40 depth=2	// L52
  hls::stream< int32_t > v41;
  #pragma HLS stream variable=v41 depth=2	// L53
  hls::stream< int32_t > v42;
  #pragma HLS stream variable=v42 depth=2	// L54
  pe_r2_0(v40, v39, v41, v42);	// L55
}

