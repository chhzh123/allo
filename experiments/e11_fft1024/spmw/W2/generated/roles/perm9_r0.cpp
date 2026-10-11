
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
void perm9_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3
) {	// L2
  float vr[256];	// L6
  for (int v5 = 0; v5 < 256; v5++) {	// L7
    vr[v5] = (float)0.000000;	// L7
  }
  float vi[256];	// L11
  for (int v7 = 0; v7 < 256; v7++) {	// L12
    vi[v7] = (float)0.000000;	// L12
  }
  float ur[256];	// L16
  for (int v9 = 0; v9 < 256; v9++) {	// L17
    ur[v9] = (float)0.000000;	// L17
  }
  float ui[256];	// L21
  for (int v11 = 0; v11 < 256; v11++) {	// L22
    ui[v11] = (float)0.000000;	// L22
  }
  l_S_t_0_t: for (int t = 0; t < 17151; t++) {	// L23
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
    ap_int<33> v15 = t;	// L33
    ap_int<33> v16 = v15 & 255;	// L34
    int32_t v17 = v16;	// L35
    int32_t c;	// L36
    c = v17;	// L37
    int32_t v19 = c;	// L38
    int v20 = v19;	// L39
    float v21 = vr[v20];	// L40
    float v1r;	// L41
    v1r = v21;	// L42
    int32_t v23 = c;	// L43
    int v24 = v23;	// L44
    float v25 = vi[v24];	// L45
    float v1i;	// L46
    v1i = v25;	// L47
    float v27 = v14[0];	// L48
    int32_t v28 = c;	// L49
    int v29 = v28;	// L50
    vr[v29] = v27;	// L51
    float v30 = v14[1];	// L52
    int32_t v31 = c;	// L53
    int v32 = v31;	// L54
    vi[v32] = v30;	// L55
    ap_int<34> v33 = t;	// L56
    ap_int<34> v34 = v33 + 1;	// L60
    ap_int<34> v35 = v34 >> 8;	// L64
    ap_int<34> v36 = v35 & 1;	// L68
    int32_t v37 = v36;	// L69
    int32_t sw;	// L70
    sw = v37;	// L71
    float v39 = v13[0];	// L72
    float u2r;	// L73
    u2r = v39;	// L74
    float v41 = v13[1];	// L75
    float u2i;	// L76
    u2i = v41;	// L77
    float v43 = v1r;	// L78
    float v2r;	// L79
    v2r = v43;	// L80
    float v45 = v1i;	// L81
    float v2i;	// L82
    v2i = v45;	// L83
    int32_t v47 = sw;	// L84
    bool v48 = v47 == 1;	// L87
    if (v48) {	// L88
      float v49 = v1r;	// L89
      u2r = v49;	// L90
      float v50 = v1i;	// L91
      u2i = v50;	// L92
      float v51 = v13[0];	// L93
      v2r = v51;	// L94
      float v52 = v13[1];	// L95
      v2i = v52;	// L96
    }
    float p[2];	// L101
    for (int v54 = 0; v54 < 2; v54++) {	// L102
      p[v54] = (float)0.000000;	// L102
    }
    float q[2];	// L106
    for (int v56 = 0; v56 < 2; v56++) {	// L107
      q[v56] = (float)0.000000;	// L107
    }
    int32_t v57 = c;	// L108
    int v58 = v57;	// L109
    float v59 = ur[v58];	// L110
    p[0] = v59;	// L111
    int32_t v60 = c;	// L112
    int v61 = v60;	// L113
    float v62 = ui[v61];	// L114
    p[1] = v62;	// L115
    float v63 = u2r;	// L116
    int32_t v64 = c;	// L117
    int v65 = v64;	// L118
    ur[v65] = v63;	// L119
    float v66 = u2i;	// L120
    int32_t v67 = c;	// L121
    int v68 = v67;	// L122
    ui[v68] = v66;	// L123
    float v69 = v2r;	// L124
    q[0] = v69;	// L125
    float v70 = v2i;	// L126
    q[1] = v70;	// L127
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

