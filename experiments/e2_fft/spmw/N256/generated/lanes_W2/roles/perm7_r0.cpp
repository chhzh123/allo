
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
void perm7_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3
) {	// L2
  float vr[64];	// L8
  for (int v5 = 0; v5 < 64; v5++) {	// L9
    vr[v5] = (float)0.000000;	// L9
  }
  float vi[64];	// L10
  for (int v7 = 0; v7 < 64; v7++) {	// L11
    vi[v7] = (float)0.000000;	// L11
  }
  float ur[64];	// L12
  for (int v9 = 0; v9 < 64; v9++) {	// L13
    ur[v9] = (float)0.000000;	// L13
  }
  float ui[64];	// L14
  for (int v11 = 0; v11 < 64; v11++) {	// L15
    ui[v11] = (float)0.000000;	// L15
  }
  l_S_t_0_t: for (int t = 0; t < 4287; t++) {	// L16
    float v13[2];
    {
      hls::vector< float, 2 > _vec = v0.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v13[_iv0] = _vec[_iv0];
      }
    }	// L17
    float v14[2];
    {
      hls::vector< float, 2 > _vec = v1.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v14[_iv0] = _vec[_iv0];
      }
    }	// L18
    ap_int<33> v15 = t;	// L19
    ap_int<33> v16 = v15 & 63;	// L20
    int32_t v17 = v16;	// L21
    int32_t c;	// L22
    c = v17;	// L23
    int32_t v19 = c;	// L24
    int v20 = v19;	// L25
    float v21 = vr[v20];	// L26
    float v1r;	// L27
    v1r = v21;	// L28
    int32_t v23 = c;	// L29
    int v24 = v23;	// L30
    float v25 = vi[v24];	// L31
    float v1i;	// L32
    v1i = v25;	// L33
    float v27 = v14[0];	// L34
    int32_t v28 = c;	// L35
    int v29 = v28;	// L36
    vr[v29] = v27;	// L37
    float v30 = v14[1];	// L38
    int32_t v31 = c;	// L39
    int v32 = v31;	// L40
    vi[v32] = v30;	// L41
    ap_int<34> v33 = t;	// L42
    ap_int<34> v34 = v33 + 1;	// L43
    ap_int<34> v35 = v34 >> 6;	// L44
    ap_int<34> v36 = v35 & 1;	// L45
    int32_t v37 = v36;	// L46
    int32_t sw;	// L47
    sw = v37;	// L48
    float v39 = v13[0];	// L49
    float u2r;	// L50
    u2r = v39;	// L51
    float v41 = v13[1];	// L52
    float u2i;	// L53
    u2i = v41;	// L54
    float v43 = v1r;	// L55
    float v2r;	// L56
    v2r = v43;	// L57
    float v45 = v1i;	// L58
    float v2i;	// L59
    v2i = v45;	// L60
    int32_t v47 = sw;	// L61
    bool v48 = v47 == 1;	// L62
    if (v48) {	// L63
      float v49 = v1r;	// L64
      u2r = v49;	// L65
      float v50 = v1i;	// L66
      u2i = v50;	// L67
      float v51 = v13[0];	// L68
      v2r = v51;	// L69
      float v52 = v13[1];	// L70
      v2i = v52;	// L71
    }
    float p[2];	// L73
    for (int v54 = 0; v54 < 2; v54++) {	// L74
      p[v54] = (float)0.000000;	// L74
    }
    float q[2];	// L75
    for (int v56 = 0; v56 < 2; v56++) {	// L76
      q[v56] = (float)0.000000;	// L76
    }
    int32_t v57 = c;	// L77
    int v58 = v57;	// L78
    float v59 = ur[v58];	// L79
    p[0] = v59;	// L80
    int32_t v60 = c;	// L81
    int v61 = v60;	// L82
    float v62 = ui[v61];	// L83
    p[1] = v62;	// L84
    float v63 = u2r;	// L85
    int32_t v64 = c;	// L86
    int v65 = v64;	// L87
    ur[v65] = v63;	// L88
    float v66 = u2i;	// L89
    int32_t v67 = c;	// L90
    int v68 = v67;	// L91
    ui[v68] = v66;	// L92
    float v69 = v2r;	// L93
    q[0] = v69;	// L94
    float v70 = v2i;	// L95
    q[1] = v70;	// L96
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = p[_iv0];
      }
      v2.write(_vec);
    }	// L97
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = q[_iv0];
      }
      v3.write(_vec);
    }	// L98
  }
}

/// This is top function.
void top(

) {	// L102
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v71;
  #pragma HLS stream variable=v71 depth=8	// L103
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v72;
  #pragma HLS stream variable=v72 depth=8	// L104
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v73;
  #pragma HLS stream variable=v73 depth=8	// L105
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v74;
  #pragma HLS stream variable=v74 depth=8	// L106
  perm7_r0_0(v71, v73, v72, v74);	// L107
}

