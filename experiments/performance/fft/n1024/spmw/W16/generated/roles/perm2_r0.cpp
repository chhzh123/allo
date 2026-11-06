
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
void perm2_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3
) {	// L2
  float vr[8];	// L6
  for (int v5 = 0; v5 < 8; v5++) {	// L7
    vr[v5] = (float)0.000000;	// L7
  }
  float vi[8];	// L11
  for (int v7 = 0; v7 < 8; v7++) {	// L12
    vi[v7] = (float)0.000000;	// L12
  }
  float ur[8];	// L16
  for (int v9 = 0; v9 < 8; v9++) {	// L17
    ur[v9] = (float)0.000000;	// L17
  }
  float ui[8];	// L21
  for (int v11 = 0; v11 < 8; v11++) {	// L22
    ui[v11] = (float)0.000000;	// L22
  }
  l_S_t_0_t: for (int t = 0; t < 2143; t++) {	// L23
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
    ap_int<33> v16 = v15 & 7;	// L34
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
    ap_int<34> v34 = v33 >> 3;	// L64
    ap_int<34> v35 = v34 & 1;	// L68
    int32_t v36 = v35;	// L69
    int32_t sw;	// L70
    sw = v36;	// L71
    float v38 = v13[0];	// L72
    float u2r;	// L73
    u2r = v38;	// L74
    float v40 = v13[1];	// L75
    float u2i;	// L76
    u2i = v40;	// L77
    float v42 = v1r;	// L78
    float v2r;	// L79
    v2r = v42;	// L80
    float v44 = v1i;	// L81
    float v2i;	// L82
    v2i = v44;	// L83
    int32_t v46 = sw;	// L84
    bool v47 = v46 == 1;	// L87
    if (v47) {	// L88
      float v48 = v1r;	// L89
      u2r = v48;	// L90
      float v49 = v1i;	// L91
      u2i = v49;	// L92
      float v50 = v13[0];	// L93
      v2r = v50;	// L94
      float v51 = v13[1];	// L95
      v2i = v51;	// L96
    }
    float p[2];	// L101
    for (int v53 = 0; v53 < 2; v53++) {	// L102
      p[v53] = (float)0.000000;	// L102
    }
    float q[2];	// L106
    for (int v55 = 0; v55 < 2; v55++) {	// L107
      q[v55] = (float)0.000000;	// L107
    }
    int32_t v56 = c;	// L108
    int v57 = v56;	// L109
    float v58 = ur[v57];	// L110
    p[0] = v58;	// L111
    int32_t v59 = c;	// L112
    int v60 = v59;	// L113
    float v61 = ui[v60];	// L114
    p[1] = v61;	// L115
    float v62 = u2r;	// L116
    int32_t v63 = c;	// L117
    int v64 = v63;	// L118
    ur[v64] = v62;	// L119
    float v65 = u2i;	// L120
    int32_t v66 = c;	// L121
    int v67 = v66;	// L122
    ui[v67] = v65;	// L123
    float v68 = v2r;	// L124
    q[0] = v68;	// L125
    float v69 = v2i;	// L126
    q[1] = v69;	// L127
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

