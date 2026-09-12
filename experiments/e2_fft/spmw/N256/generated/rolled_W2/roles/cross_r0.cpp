
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
float _st_tw[1][2][2] = {1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00};	// L2
void cross_r0_0(
  hls::stream< int32_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3,
  hls::stream< hls::vector< float, 2 > >& v4
) {	// L3
  // placeholder for const float _st_tw	// L10
  int32_t v6 = v0.read();	// L11
  int32_t _st__pid0;	// L12
  _st__pid0 = v6;	// L13
  int32_t v8 = v1.read();	// L14
  int32_t _st__pid1;	// L15
  _st__pid1 = v8;	// L16
  int32_t v10 = _st__pid0;	// L17
  int32_t t;	// L18
  t = v10;	// L19
  int32_t v12 = _st__pid1;	// L20
  int32_t ell;	// L21
  ell = v12;	// L22
  int32_t v14 = t;	// L23
  ap_int<33> v15 = v14;	// L24
  ap_int<33> v16 = v15 - 1;	// L25
  int32_t v17 = v16;	// L26
  int32_t k;	// L27
  k = v17;	// L28
  int32_t v19 = t;	// L29
  ap_int<33> v20 = v19;	// L30
  ap_int<33> v21 = 1 - v20;	// L31
  int32_t v22 = v21;	// L32
  int32_t d;	// L33
  d = v22;	// L34
  int32_t v24 = ell;	// L35
  int32_t v25 = d;	// L36
  int32_t v26 = v24 >> v25;	// L37
  int32_t v27 = v26 & 1;	// L38
  int32_t half;	// L39
  half = v27;	// L40
  int32_t v29 = ell;	// L41
  int32_t v30 = d;	// L42
  int32_t v31 = 1 << v30;	// L43
  ap_int<33> v32 = v31;	// L44
  ap_int<33> v33 = v32 - 1;	// L45
  ap_int<33> v34 = v29;	// L46
  ap_int<33> v35 = v34 & v33;	// L47
  int32_t v36 = v35;	// L48
  int32_t c;	// L49
  c = v36;	// L50
  l_S__r_0__r: for (int _r = 0; _r < 4352; _r++) {	// L51
    float v39[2];
    {
      hls::vector< float, 2 > _vec = v2.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v39[_iv0] = _vec[_iv0];
      }
    }	// L52
    float v40[2];
    {
      hls::vector< float, 2 > _vec = v3.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v40[_iv0] = _vec[_iv0];
      }
    }	// L53
    float o[2];	// L54
    for (int v42 = 0; v42 < 2; v42++) {	// L55
      o[v42] = (float)0.000000;	// L55
    }
    int32_t v43 = half;	// L56
    bool v44 = v43 == 0;	// L57
    if (v44) {	// L58
      float v45 = v39[0];	// L59
      float v46 = v40[0];	// L60
      float v47 = v45 + v46;	// L61
      #pragma HLS bind_op variable=v47 op=fadd impl=fabric
      o[0] = v47;	// L62
      float v48 = v39[1];	// L63
      float v49 = v40[1];	// L64
      float v50 = v48 + v49;	// L65
      #pragma HLS bind_op variable=v50 op=fadd impl=fabric
      o[1] = v50;	// L66
    } else {
      float v51 = v40[0];	// L68
      float v52 = v39[0];	// L69
      float v53 = v51 - v52;	// L70
      #pragma HLS bind_op variable=v53 op=fsub impl=fabric
      float dr;	// L71
      dr = v53;	// L72
      float v55 = v40[1];	// L73
      float v56 = v39[1];	// L74
      float v57 = v55 - v56;	// L75
      #pragma HLS bind_op variable=v57 op=fsub impl=fabric
      float di;	// L76
      di = v57;	// L77
      int32_t v59 = k;	// L78
      int v60 = v59;	// L79
      int32_t v61 = c;	// L80
      int v62 = v61;	// L81
      float v63 = _st_tw[v60][v62][0];	// L82
      float wr;	// L83
      wr = v63;	// L84
      int32_t v65 = k;	// L85
      int v66 = v65;	// L86
      int32_t v67 = c;	// L87
      int v68 = v67;	// L88
      float v69 = _st_tw[v66][v68][1];	// L89
      float wi;	// L90
      wi = v69;	// L91
      float v71 = dr;	// L92
      float v72 = wr;	// L93
      float v73 = v71 * v72;	// L94
      float v74 = di;	// L95
      float v75 = wi;	// L96
      float v76 = v74 * v75;	// L97
      float v77 = v73 - v76;	// L98
      #pragma HLS bind_op variable=v77 op=fsub impl=fabric
      o[0] = v77;	// L99
      float v78 = dr;	// L100
      float v79 = wi;	// L101
      float v80 = v78 * v79;	// L102
      float v81 = di;	// L103
      float v82 = wr;	// L104
      float v83 = v81 * v82;	// L105
      float v84 = v80 + v83;	// L106
      #pragma HLS bind_op variable=v84 op=fadd impl=fabric
      o[1] = v84;	// L107
    }
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = o[_iv0];
      }
      v4.write(_vec);
    }	// L109
  }
}

/// This is top function.
void top(

) {	// L113
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v85;
  #pragma HLS stream variable=v85 depth=8	// L114
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v86;
  #pragma HLS stream variable=v86 depth=8	// L115
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v87;
  #pragma HLS stream variable=v87 depth=8	// L116
  hls::stream< int32_t > v88;
  #pragma HLS stream variable=v88 depth=1	// L117
  hls::stream< int32_t > v89;
  #pragma HLS stream variable=v89 depth=1	// L118
  cross_r0_0(v88, v89, v85, v87, v86);	// L119
}

