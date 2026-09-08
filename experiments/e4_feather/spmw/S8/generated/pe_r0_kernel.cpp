
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
void pe_r0_0(
  hls::stream< hls::vector< int8_t, 2 > >& v0,
  hls::stream< hls::vector< int8_t, 16 > >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int32_t >& v4,
  hls::stream< int32_t >& v5
) {	// L2
  int8_t v6[2];
  {
    hls::vector< int8_t, 2 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v6[_iv0] = _vec[_iv0];
    }
  }	// L3
  int8_t v7[16];
  {
    hls::vector< int8_t, 16 > _vec = v1.read();
    for (int _iv0 = 0; _iv0 < 16; ++_iv0) {
      v7[_iv0] = _vec[_iv0];
    }
  }	// L4
  l_S_i_0_i: for (int i = 0; i < 8; i++) {	// L5
    int32_t v9 = v2.read();	// L6
    int32_t p0;	// L7
    p0 = v9;	// L8
    int32_t v11 = v3.read();	// L9
    int32_t p1;	// L10
    p1 = v11;	// L11
    int8_t v13 = v6[0];	// L12
    int32_t v14 = v13;	// L13
    int32_t x0;	// L14
    x0 = v14;	// L15
    int8_t v16 = v6[1];	// L16
    int32_t v17 = v16;	// L17
    int32_t x1;	// L18
    x1 = v17;	// L19
    int8_t v19 = v7[i];	// L20
    int32_t v20 = v19;	// L21
    int32_t w0;	// L22
    w0 = v20;	// L23
    int8_t v22 = v7[(i + 8)];	// L24
    int32_t v23 = v22;	// L25
    int32_t w1;	// L26
    w1 = v23;	// L27
    int32_t v25 = p0;	// L28
    int32_t v26 = x0;	// L29
    int32_t v27 = w0;	// L30
    int64_t v28 = v26;	// L31
    int64_t v29 = v27;	// L32
    int64_t v30 = v28 * v29;	// L33
    ap_int<65> v31 = v25;	// L34
    ap_int<65> v32 = v30;	// L35
    ap_int<65> v33 = v31 + v32;	// L36
    v4.write(v33);	// L37
    int32_t v34 = p1;	// L38
    int32_t v35 = x1;	// L39
    int32_t v36 = w1;	// L40
    int64_t v37 = v35;	// L41
    int64_t v38 = v36;	// L42
    int64_t v39 = v37 * v38;	// L43
    ap_int<65> v40 = v34;	// L44
    ap_int<65> v41 = v39;	// L45
    ap_int<65> v42 = v40 + v41;	// L46
    v5.write(v42);	// L47
  }
}

/// This is top function.
void top(

) {	// L51
  #pragma HLS dataflow
  // Stream of vectors: each vector packs int8_t array[16] into hls::vector<int8_t, 16>
  hls::stream< hls::vector< int8_t, 16 > > v43;
  #pragma HLS stream variable=v43 depth=2	// L52
  // Stream of vectors: each vector packs int8_t array[2] into hls::vector<int8_t, 2>
  hls::stream< hls::vector< int8_t, 2 > > v44;
  #pragma HLS stream variable=v44 depth=2	// L53
  hls::stream< int32_t > v45;
  #pragma HLS stream variable=v45 depth=2	// L54
  hls::stream< int32_t > v46;
  #pragma HLS stream variable=v46 depth=2	// L55
  hls::stream< int32_t > v47;
  #pragma HLS stream variable=v47 depth=2	// L56
  hls::stream< int32_t > v48;
  #pragma HLS stream variable=v48 depth=2	// L57
  pe_r0_0(v44, v43, v45, v47, v46, v48);	// L58
}

