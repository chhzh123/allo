
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
void switch_r4_0(
  hls::stream< hls::vector< int8_t, 2 > >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int32_t >& v4
) {	// L2
  int8_t v5[2];
  {
    hls::vector< int8_t, 2 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v5[_iv0] = _vec[_iv0];
    }
  }	// L6
  int8_t v6 = v5[0];	// L7
  int32_t v7 = v6;	// L8
  int32_t c;	// L9
  c = v7;	// L10
  l_S__i_0__i: for (int _i = 0; _i < 4; _i++) {	// L11
    int32_t v10 = v1.read();	// L12
    int32_t l;	// L13
    l = v10;	// L14
    int32_t v12 = v2.read();	// L15
    int32_t r;	// L16
    r = v12;	// L17
    int32_t v14 = l;	// L18
    int32_t ol;	// L19
    ol = v14;	// L20
    int32_t v16 = r;	// L21
    int32_t orr;	// L22
    orr = v16;	// L23
    int32_t v18 = c;	// L24
    bool v19 = v18 == 1;	// L25
    if (v19) {	// L26
      int32_t v20 = l;	// L27
      int32_t v21 = r;	// L28
      ap_int<33> v22 = v20;	// L29
      ap_int<33> v23 = v21;	// L30
      ap_int<33> v24 = v22 + v23;	// L31
      int32_t v25 = v24;	// L32
      orr = v25;	// L33
    } else {
      int32_t v26 = c;	// L35
      bool v27 = v26 == 2;	// L36
      if (v27) {	// L37
        int32_t v28 = l;	// L38
        int32_t v29 = r;	// L39
        ap_int<33> v30 = v28;	// L40
        ap_int<33> v31 = v29;	// L41
        ap_int<33> v32 = v30 + v31;	// L42
        int32_t v33 = v32;	// L43
        ol = v33;	// L44
      } else {
        int32_t v34 = c;	// L46
        bool v35 = v34 == 3;	// L47
        if (v35) {	// L48
          int32_t v36 = r;	// L49
          ol = v36;	// L50
          int32_t v37 = l;	// L51
          orr = v37;	// L52
        }
      }
    }
    int32_t v38 = ol;	// L56
    v3.write(v38);	// L57
    int32_t v39 = orr;	// L58
    v4.write(v39);	// L59
  }
}

/// This is top function.
void top(

) {	// L63
  #pragma HLS dataflow
  // Stream of vectors: each vector packs int8_t array[2] into hls::vector<int8_t, 2>
  hls::stream< hls::vector< int8_t, 2 > > v40;
  #pragma HLS stream variable=v40 depth=2	// L64
  hls::stream< int32_t > v41;
  #pragma HLS stream variable=v41 depth=2	// L65
  hls::stream< int32_t > v42;
  #pragma HLS stream variable=v42 depth=2	// L66
  hls::stream< int32_t > v43;
  #pragma HLS stream variable=v43 depth=2	// L67
  hls::stream< int32_t > v44;
  #pragma HLS stream variable=v44 depth=2	// L68
  switch_r4_0(v40, v41, v42, v43, v44);	// L69
}

