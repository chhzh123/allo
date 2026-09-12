
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
void perm1_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3
) {	// L2
  float vr[32];	// L8
  for (int v5 = 0; v5 < 32; v5++) {	// L9
    vr[v5] = (float)0.000000;	// L9
  }
  float vi[32];	// L10
  for (int v7 = 0; v7 < 32; v7++) {	// L11
    vi[v7] = (float)0.000000;	// L11
  }
  float ur[32];	// L12
  for (int v9 = 0; v9 < 32; v9++) {	// L13
    ur[v9] = (float)0.000000;	// L13
  }
  float ui[32];	// L14
  for (int v11 = 0; v11 < 32; v11++) {	// L15
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
    ap_int<33> v16 = v15 & 31;	// L20
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
    ap_int<34> v34 = v33 >> 5;	// L43
    ap_int<34> v35 = v34 & 1;	// L44
    int32_t v36 = v35;	// L45
    int32_t sw;	// L46
    sw = v36;	// L47
    float v38 = v13[0];	// L48
    float u2r;	// L49
    u2r = v38;	// L50
    float v40 = v13[1];	// L51
    float u2i;	// L52
    u2i = v40;	// L53
    float v42 = v1r;	// L54
    float v2r;	// L55
    v2r = v42;	// L56
    float v44 = v1i;	// L57
    float v2i;	// L58
    v2i = v44;	// L59
    int32_t v46 = sw;	// L60
    bool v47 = v46 == 1;	// L61
    if (v47) {	// L62
      float v48 = v1r;	// L63
      u2r = v48;	// L64
      float v49 = v1i;	// L65
      u2i = v49;	// L66
      float v50 = v13[0];	// L67
      v2r = v50;	// L68
      float v51 = v13[1];	// L69
      v2i = v51;	// L70
    }
    float p[2];	// L72
    for (int v53 = 0; v53 < 2; v53++) {	// L73
      p[v53] = (float)0.000000;	// L73
    }
    float q[2];	// L74
    for (int v55 = 0; v55 < 2; v55++) {	// L75
      q[v55] = (float)0.000000;	// L75
    }
    int32_t v56 = c;	// L76
    int v57 = v56;	// L77
    float v58 = ur[v57];	// L78
    p[0] = v58;	// L79
    int32_t v59 = c;	// L80
    int v60 = v59;	// L81
    float v61 = ui[v60];	// L82
    p[1] = v61;	// L83
    float v62 = u2r;	// L84
    int32_t v63 = c;	// L85
    int v64 = v63;	// L86
    ur[v64] = v62;	// L87
    float v65 = u2i;	// L88
    int32_t v66 = c;	// L89
    int v67 = v66;	// L90
    ui[v67] = v65;	// L91
    float v68 = v2r;	// L92
    q[0] = v68;	// L93
    float v69 = v2i;	// L94
    q[1] = v69;	// L95
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = p[_iv0];
      }
      v2.write(_vec);
    }	// L96
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = q[_iv0];
      }
      v3.write(_vec);
    }	// L97
  }
}

/// This is top function.
void top(

) {	// L101
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v70;
  #pragma HLS stream variable=v70 depth=8	// L102
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v71;
  #pragma HLS stream variable=v71 depth=8	// L103
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v72;
  #pragma HLS stream variable=v72 depth=8	// L104
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v73;
  #pragma HLS stream variable=v73 depth=8	// L105
  perm1_r0_0(v70, v72, v71, v73);	// L106
}

