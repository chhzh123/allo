
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>
using namespace std;
float _st_tw[4][8][2] = {1.000000e+00, 0.000000e+00, 9.238795e-01, -3.826834e-01, 7.071068e-01, -7.071068e-01, 3.826834e-01, -9.238795e-01, 6.123234e-17, -1.000000e+00, -3.826834e-01, -9.238795e-01, -7.071068e-01, -7.071068e-01, -9.238795e-01, -3.826834e-01, 1.000000e+00, 0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01, 1.000000e+00, 0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01, 1.000000e+00, 0.000000e+00, 6.123234e-17, -1.000000e+00, 1.000000e+00, 0.000000e+00, 6.123234e-17, -1.000000e+00, 1.000000e+00, 0.000000e+00, 6.123234e-17, -1.000000e+00, 1.000000e+00, 0.000000e+00, 6.123234e-17, -1.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00};	// L2
int32_t _st_sel[4][8] = {1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1};	// L3
/// This is top function.
void tail_general_r28_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3
) {	// L4
  // placeholder for const float _st_tw	// L5
  // placeholder for const int32_t _st_sel	// L6
  int32_t k;	// L9
  k = 3;	// L10
  int32_t m;	// L13
  m = 4;	// L14
  l_S__t_0__t: for (int _t = 0; _t < 2143; _t++) {	// L15
    float v9[2];
    {
      hls::vector< float, 2 > _vec = v0.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v9[_iv0] = _vec[_iv0];
      }
    }	// L16
    float v10[2];
    {
      hls::vector< float, 2 > _vec = v2.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v10[_iv0] = _vec[_iv0];
      }
    }	// L17
    float v11 = v9[0];	// L18
    float v12 = v10[0];	// L19
    float v13 = v11 - v12;	// L20
    #pragma HLS bind_op variable=v13 op=fsub impl=fabric
    float dr;	// L21
    dr = v13;	// L22
    float v15 = v9[1];	// L23
    float v16 = v10[1];	// L24
    float v17 = v15 - v16;	// L25
    #pragma HLS bind_op variable=v17 op=fsub impl=fabric
    float di;	// L26
    di = v17;	// L27
    int32_t v19 = k;	// L28
    int v20 = v19;	// L29
    int32_t v21 = m;	// L30
    int v22 = v21;	// L31
    float v23 = _st_tw[v20][v22][0];	// L35
    float wr;	// L36
    wr = v23;	// L37
    int32_t v25 = k;	// L38
    int v26 = v25;	// L39
    int32_t v27 = m;	// L40
    int v28 = v27;	// L41
    float v29 = _st_tw[v26][v28][1];	// L45
    float wi;	// L46
    wi = v29;	// L47
    int32_t v31 = k;	// L48
    int v32 = v31;	// L49
    int32_t v33 = m;	// L50
    int v34 = v33;	// L51
    int32_t v35 = _st_sel[v32][v34];	// L52
    int32_t rot;	// L53
    rot = v35;	// L54
    float p[2];	// L58
    for (int v38 = 0; v38 < 2; v38++) {	// L59
      p[v38] = (float)0.000000;	// L59
    }
    float q[2];	// L63
    for (int v40 = 0; v40 < 2; v40++) {	// L64
      q[v40] = (float)0.000000;	// L64
    }
    float v41 = v9[0];	// L65
    float v42 = v10[0];	// L66
    float v43 = v41 + v42;	// L67
    #pragma HLS bind_op variable=v43 op=fadd impl=fabric
    p[0] = v43;	// L68
    float v44 = v9[1];	// L69
    float v45 = v10[1];	// L70
    float v46 = v44 + v45;	// L71
    #pragma HLS bind_op variable=v46 op=fadd impl=fabric
    p[1] = v46;	// L72
    float v47 = dr;	// L73
    float v48 = wr;	// L74
    float v49 = v47 * v48;	// L75
    float v50 = di;	// L76
    float v51 = wi;	// L77
    float v52 = v50 * v51;	// L78
    float v53 = v49 - v52;	// L79
    #pragma HLS bind_op variable=v53 op=fsub impl=fabric
    q[0] = v53;	// L80
    float v54 = dr;	// L81
    float v55 = wi;	// L82
    float v56 = v54 * v55;	// L83
    float v57 = di;	// L84
    float v58 = wr;	// L85
    float v59 = v57 * v58;	// L86
    float v60 = v56 + v59;	// L87
    #pragma HLS bind_op variable=v60 op=fadd impl=fabric
    q[1] = v60;	// L88
    int32_t v61 = rot;	// L89
    bool v62 = v61 == 1;	// L92
    if (v62) {	// L93
      float v63 = dr;	// L94
      q[0] = v63;	// L95
      float v64 = di;	// L96
      q[1] = v64;	// L97
    }
    int32_t v65 = rot;	// L99
    bool v66 = v65 == 2;	// L102
    if (v66) {	// L103
      float v67 = di;	// L104
      q[0] = v67;	// L105
      float v68 = dr;	// L106
      float v69 = (float)0.000000 - v68;	// L109
      #pragma HLS bind_op variable=v69 op=fsub impl=fabric
      q[1] = v69;	// L110
    }
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = p[_iv0];
      }
      v1.write(_vec);
    }	// L112
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = q[_iv0];
      }
      v3.write(_vec);
    }	// L113
  }
}

