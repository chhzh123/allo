
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
void perm5_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3
) {	// L2
  float vr[2];	// L7
  for (int v5 = 0; v5 < 2; v5++) {	// L8
    vr[v5] = (float)0.000000;	// L8
  }
  float vi[2];	// L9
  for (int v7 = 0; v7 < 2; v7++) {	// L10
    vi[v7] = (float)0.000000;	// L10
  }
  float ur[2];	// L11
  for (int v9 = 0; v9 < 2; v9++) {	// L12
    ur[v9] = (float)0.000000;	// L12
  }
  float ui[2];	// L13
  for (int v11 = 0; v11 < 2; v11++) {	// L14
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
    ap_int<33> v15 = t;	// L18
    ap_int<33> v16 = v15 & 1;	// L19
    int32_t v17 = v16;	// L20
    int32_t c;	// L21
    c = v17;	// L22
    int32_t v19 = c;	// L23
    int v20 = v19;	// L24
    float v21 = vr[v20];	// L25
    float v1r;	// L26
    v1r = v21;	// L27
    int32_t v23 = c;	// L28
    int v24 = v23;	// L29
    float v25 = vi[v24];	// L30
    float v1i;	// L31
    v1i = v25;	// L32
    float v27 = v14[0];	// L33
    int32_t v28 = c;	// L34
    int v29 = v28;	// L35
    vr[v29] = v27;	// L36
    float v30 = v14[1];	// L37
    int32_t v31 = c;	// L38
    int v32 = v31;	// L39
    vi[v32] = v30;	// L40
    ap_int<34> v33 = t;	// L41
    ap_int<34> v34 = v33 >> 1;	// L42
    ap_int<34> v35 = v34 & 1;	// L43
    int32_t v36 = v35;	// L44
    int32_t sw;	// L45
    sw = v36;	// L46
    float v38 = v13[0];	// L47
    float u2r;	// L48
    u2r = v38;	// L49
    float v40 = v13[1];	// L50
    float u2i;	// L51
    u2i = v40;	// L52
    float v42 = v1r;	// L53
    float v2r;	// L54
    v2r = v42;	// L55
    float v44 = v1i;	// L56
    float v2i;	// L57
    v2i = v44;	// L58
    int32_t v46 = sw;	// L59
    bool v47 = v46 == 1;	// L60
    if (v47) {	// L61
      float v48 = v1r;	// L62
      u2r = v48;	// L63
      float v49 = v1i;	// L64
      u2i = v49;	// L65
      float v50 = v13[0];	// L66
      v2r = v50;	// L67
      float v51 = v13[1];	// L68
      v2i = v51;	// L69
    }
    float p[2];	// L71
    for (int v53 = 0; v53 < 2; v53++) {	// L72
      p[v53] = (float)0.000000;	// L72
    }
    float q[2];	// L73
    for (int v55 = 0; v55 < 2; v55++) {	// L74
      q[v55] = (float)0.000000;	// L74
    }
    int32_t v56 = c;	// L75
    int v57 = v56;	// L76
    float v58 = ur[v57];	// L77
    p[0] = v58;	// L78
    int32_t v59 = c;	// L79
    int v60 = v59;	// L80
    float v61 = ui[v60];	// L81
    p[1] = v61;	// L82
    float v62 = u2r;	// L83
    int32_t v63 = c;	// L84
    int v64 = v63;	// L85
    ur[v64] = v62;	// L86
    float v65 = u2i;	// L87
    int32_t v66 = c;	// L88
    int v67 = v66;	// L89
    ui[v67] = v65;	// L90
    float v68 = v2r;	// L91
    q[0] = v68;	// L92
    float v69 = v2i;	// L93
    q[1] = v69;	// L94
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = p[_iv0];
      }
      v2.write(_vec);
    }	// L95
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = q[_iv0];
      }
      v3.write(_vec);
    }	// L96
  }
}

/// This is top function.
void top(

) {	// L100
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v70;
  #pragma HLS stream variable=v70 depth=8	// L101
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v71;
  #pragma HLS stream variable=v71 depth=8	// L102
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v72;
  #pragma HLS stream variable=v72 depth=8	// L103
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v73;
  #pragma HLS stream variable=v73 depth=8	// L104
  perm5_r0_0(v70, v72, v71, v73);	// L105
}

