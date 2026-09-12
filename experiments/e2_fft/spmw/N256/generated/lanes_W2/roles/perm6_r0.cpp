
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
void perm6_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3
) {	// L2
  float vr[1];	// L7
  for (int v5 = 0; v5 < 1; v5++) {	// L8
    vr[v5] = (float)0.000000;	// L8
  }
  float vi[1];	// L9
  for (int v7 = 0; v7 < 1; v7++) {	// L10
    vi[v7] = (float)0.000000;	// L10
  }
  float ur[1];	// L11
  for (int v9 = 0; v9 < 1; v9++) {	// L12
    ur[v9] = (float)0.000000;	// L12
  }
  float ui[1];	// L13
  for (int v11 = 0; v11 < 1; v11++) {	// L14
    ui[v11] = (float)0.000000;	// L14
  }
  l_S_t_0_t: for (int t = 0; t < 4287; t++) {	// L15
    float v13[2];
    {
      hls::vector< float, 2 > _vec = v0.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v13[_iv0] = _vec[_iv0];
      }
    }	// L16
    float v14[2];
    {
      hls::vector< float, 2 > _vec = v1.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v14[_iv0] = _vec[_iv0];
      }
    }	// L17
    int32_t c;	// L18
    c = 0;	// L19
    int32_t v16 = c;	// L20
    int v17 = v16;	// L21
    float v18 = vr[v17];	// L22
    float v1r;	// L23
    v1r = v18;	// L24
    int32_t v20 = c;	// L25
    int v21 = v20;	// L26
    float v22 = vi[v21];	// L27
    float v1i;	// L28
    v1i = v22;	// L29
    float v24 = v14[0];	// L30
    int32_t v25 = c;	// L31
    int v26 = v25;	// L32
    vr[v26] = v24;	// L33
    float v27 = v14[1];	// L34
    int32_t v28 = c;	// L35
    int v29 = v28;	// L36
    vi[v29] = v27;	// L37
    ap_int<34> v30 = t;	// L38
    ap_int<34> v31 = v30 & 1;	// L39
    int32_t v32 = v31;	// L40
    int32_t sw;	// L41
    sw = v32;	// L42
    float v34 = v13[0];	// L43
    float u2r;	// L44
    u2r = v34;	// L45
    float v36 = v13[1];	// L46
    float u2i;	// L47
    u2i = v36;	// L48
    float v38 = v1r;	// L49
    float v2r;	// L50
    v2r = v38;	// L51
    float v40 = v1i;	// L52
    float v2i;	// L53
    v2i = v40;	// L54
    int32_t v42 = sw;	// L55
    bool v43 = v42 == 1;	// L56
    if (v43) {	// L57
      float v44 = v1r;	// L58
      u2r = v44;	// L59
      float v45 = v1i;	// L60
      u2i = v45;	// L61
      float v46 = v13[0];	// L62
      v2r = v46;	// L63
      float v47 = v13[1];	// L64
      v2i = v47;	// L65
    }
    float p[2];	// L67
    for (int v49 = 0; v49 < 2; v49++) {	// L68
      p[v49] = (float)0.000000;	// L68
    }
    float q[2];	// L69
    for (int v51 = 0; v51 < 2; v51++) {	// L70
      q[v51] = (float)0.000000;	// L70
    }
    int32_t v52 = c;	// L71
    int v53 = v52;	// L72
    float v54 = ur[v53];	// L73
    p[0] = v54;	// L74
    int32_t v55 = c;	// L75
    int v56 = v55;	// L76
    float v57 = ui[v56];	// L77
    p[1] = v57;	// L78
    float v58 = u2r;	// L79
    int32_t v59 = c;	// L80
    int v60 = v59;	// L81
    ur[v60] = v58;	// L82
    float v61 = u2i;	// L83
    int32_t v62 = c;	// L84
    int v63 = v62;	// L85
    ui[v63] = v61;	// L86
    float v64 = v2r;	// L87
    q[0] = v64;	// L88
    float v65 = v2i;	// L89
    q[1] = v65;	// L90
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = p[_iv0];
      }
      v2.write(_vec);
    }	// L91
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = q[_iv0];
      }
      v3.write(_vec);
    }	// L92
  }
}

/// This is top function.
void top(

) {	// L96
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v66;
  #pragma HLS stream variable=v66 depth=8	// L97
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v67;
  #pragma HLS stream variable=v67 depth=8	// L98
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v68;
  #pragma HLS stream variable=v68 depth=8	// L99
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v69;
  #pragma HLS stream variable=v69 depth=8	// L100
  perm6_r0_0(v66, v68, v67, v69);	// L101
}

