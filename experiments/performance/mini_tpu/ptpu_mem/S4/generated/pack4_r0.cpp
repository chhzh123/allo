
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void pack4_r0_0(
  hls::stream< int8_t >& v0,
  hls::stream< hls::vector< int8_t, 5 > >& v1,
  hls::stream< int64_t >& v2
) {	// L2
  int64_t lo;	// L6
  lo = 0;	// L7
  bool part;	// L11
  part = 0;	// L12
  bool go;	// L16
  go = 1;	// L17
  while (true) {	// L18
    #pragma HLS pipeline II=1 style=flp
    bool v6 = go;	// L19
    if (!(v6)) break;
    int8_t v7[5];
    {
      hls::vector< int8_t, 5 > _vec = v1.read();
      for (int _iv0 = 0; _iv0 < 5; ++_iv0) {
        v7[_iv0] = _vec[_iv0];
      }
    }	// L26
    v0.write(1);	// L29
    int8_t v8 = v7[0];	// L30
    int32_t v9 = v8;	// L31
    int32_t v10 = v9 & 255;	// L34
    int64_t v11 = v10;	// L35
    int64_t half_0;	// L36
    half_0 = v11;	// L37
    int8_t v13 = v7[1];	// L38
    int32_t v14 = v13;	// L39
    int32_t v15 = v14 & 255;	// L42
    int64_t v16 = v15;	// L43
    int64_t half_1;	// L44
    half_1 = v16;	// L45
    int8_t v18 = v7[2];	// L46
    int32_t v19 = v18;	// L47
    int32_t v20 = v19 & 255;	// L50
    int64_t v21 = v20;	// L51
    int64_t half_2;	// L52
    half_2 = v21;	// L53
    int8_t v23 = v7[3];	// L54
    int32_t v24 = v23;	// L55
    int32_t v25 = v24 & 255;	// L58
    int64_t v26 = v25;	// L59
    int64_t half_3;	// L60
    half_3 = v26;	// L61
    int64_t half;	// L65
    half = 0;	// L66
    int64_t v29 = half;	// L67
    int64_t v30 = half_0;	// L68
    int64_t v31 = v29 | v30;	// L73
    half = v31;	// L74
    int64_t v32 = half;	// L75
    int64_t v33 = half_1;	// L76
    int64_t v34 = v33 << 8;	// L80
    int64_t v35 = v32 | v34;	// L81
    half = v35;	// L82
    int64_t v36 = half;	// L83
    int64_t v37 = half_2;	// L84
    int64_t v38 = v37 << 16;	// L88
    int64_t v39 = v36 | v38;	// L89
    half = v39;	// L90
    int64_t v40 = half;	// L91
    int64_t v41 = half_3;	// L92
    int64_t v42 = v41 << 24;	// L96
    int64_t v43 = v40 | v42;	// L97
    half = v43;	// L98
    bool v44 = part;	// L99
    int32_t v45 = v44;	// L100
    bool v46 = v45 == 0;	// L103
    if (v46) {	// L104
      int64_t v47 = half;	// L105
      lo = v47;	// L106
      part = 1;	// L110
    } else {
      int64_t v48 = lo;	// L112
      int64_t v49 = half;	// L113
      int64_t v50 = v49 << 32;	// L117
      int64_t v51 = v48 | v50;	// L118
      v2.write(v51);	// L119
      part = 0;	// L123
    }
    int8_t v52 = v7[4];	// L125
    int32_t v53 = v52;	// L126
    int32_t fl;	// L127
    fl = v53;	// L128
    int32_t v55 = fl;	// L129
    bool v56 = v55 != 0;	// L132
    if (v56) {	// L133
      go = 0;	// L137
    }
  }
}

