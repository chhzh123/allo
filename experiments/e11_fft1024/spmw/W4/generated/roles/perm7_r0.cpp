
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
/// This is top function.
void perm7_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3
) {	// L2
  float vr[1];	// L6
  for (int v5 = 0; v5 < 1; v5++) {	// L7
    vr[v5] = (float)0.000000;	// L7
  }
  float vi[1];	// L11
  for (int v7 = 0; v7 < 1; v7++) {	// L12
    vi[v7] = (float)0.000000;	// L12
  }
  float ur[1];	// L16
  for (int v9 = 0; v9 < 1; v9++) {	// L17
    ur[v9] = (float)0.000000;	// L17
  }
  float ui[1];	// L21
  for (int v11 = 0; v11 < 1; v11++) {	// L22
    ui[v11] = (float)0.000000;	// L22
  }
  l_S_t_0_t: for (int t = 0; t < 8575; t++) {	// L23
    float v13[2];
    {
      hls::vector< float, 2 > _vec = v0.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v13[_iv0] = _vec[_iv0];
      }
    }	// L24
    float v14[2];
    {
      hls::vector< float, 2 > _vec = v2.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v14[_iv0] = _vec[_iv0];
      }
    }	// L25
    int32_t c;	// L36
    c = 0;	// L37
    int32_t v16 = c;	// L38
    int v17 = v16;	// L39
    float v18 = vr[v17];	// L40
    float v1r;	// L41
    v1r = v18;	// L42
    int32_t v20 = c;	// L43
    int v21 = v20;	// L44
    float v22 = vi[v21];	// L45
    float v1i;	// L46
    v1i = v22;	// L47
    float v24 = v14[0];	// L48
    int32_t v25 = c;	// L49
    int v26 = v25;	// L50
    vr[v26] = v24;	// L51
    float v27 = v14[1];	// L52
    int32_t v28 = c;	// L53
    int v29 = v28;	// L54
    vi[v29] = v27;	// L55
    ap_int<34> v30 = t;	// L56
    ap_int<34> v31 = v30 & 1;	// L68
    int32_t v32 = v31;	// L69
    int32_t sw;	// L70
    sw = v32;	// L71
    float v34 = v13[0];	// L72
    float u2r;	// L73
    u2r = v34;	// L74
    float v36 = v13[1];	// L75
    float u2i;	// L76
    u2i = v36;	// L77
    float v38 = v1r;	// L78
    float v2r;	// L79
    v2r = v38;	// L80
    float v40 = v1i;	// L81
    float v2i;	// L82
    v2i = v40;	// L83
    int32_t v42 = sw;	// L84
    bool v43 = v42 == 1;	// L87
    if (v43) {	// L88
      float v44 = v1r;	// L89
      u2r = v44;	// L90
      float v45 = v1i;	// L91
      u2i = v45;	// L92
      float v46 = v13[0];	// L93
      v2r = v46;	// L94
      float v47 = v13[1];	// L95
      v2i = v47;	// L96
    }
    float p[2];	// L101
    for (int v49 = 0; v49 < 2; v49++) {	// L102
      p[v49] = (float)0.000000;	// L102
    }
    float q[2];	// L106
    for (int v51 = 0; v51 < 2; v51++) {	// L107
      q[v51] = (float)0.000000;	// L107
    }
    int32_t v52 = c;	// L108
    int v53 = v52;	// L109
    float v54 = ur[v53];	// L110
    p[0] = v54;	// L111
    int32_t v55 = c;	// L112
    int v56 = v55;	// L113
    float v57 = ui[v56];	// L114
    p[1] = v57;	// L115
    float v58 = u2r;	// L116
    int32_t v59 = c;	// L117
    int v60 = v59;	// L118
    ur[v60] = v58;	// L119
    float v61 = u2i;	// L120
    int32_t v62 = c;	// L121
    int v63 = v62;	// L122
    ui[v63] = v61;	// L123
    float v64 = v2r;	// L124
    q[0] = v64;	// L125
    float v65 = v2i;	// L126
    q[1] = v65;	// L127
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = p[_iv0];
      }
      v1.write(_vec);
    }	// L128
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = q[_iv0];
      }
      v3.write(_vec);
    }	// L129
  }
}

