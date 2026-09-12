
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
float _st_tw[2][1][2] = {1.000000e+00, 0.000000e+00, 6.123234e-17, -1.000000e+00};	// L2
void delay6_r0_0(
  hls::stream< int32_t >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2
) {	// L3
  // placeholder for const float _st_tw	// L8
  int32_t v4 = v0.read();	// L9
  int32_t _st__pid0;	// L10
  _st__pid0 = v4;	// L11
  int32_t v6 = _st__pid0;	// L12
  int32_t ell;	// L13
  ell = v6;	// L14
  float ar[2];	// L15
  for (int v9 = 0; v9 < 2; v9++) {	// L16
    ar[v9] = (float)0.000000;	// L16
  }
  float ai[2];	// L17
  for (int v11 = 0; v11 < 2; v11++) {	// L18
    ai[v11] = (float)0.000000;	// L18
  }
  float br[2];	// L19
  for (int v13 = 0; v13 < 2; v13++) {	// L20
    br[v13] = (float)0.000000;	// L20
  }
  float bi[2];	// L21
  for (int v15 = 0; v15 < 2; v15++) {	// L22
    bi[v15] = (float)0.000000;	// L22
  }
  l_S__b_0__b: for (int _b = 0; _b < 2176; _b++) {	// L23
    l_S_h_0_h: for (int h = 0; h < 2; h++) {	// L24
      l_S_c_0_c: for (int c = 0; c < 2; c++) {	// L25
        float v19[2];
        {
          hls::vector< float, 2 > _vec = v1.read();
          for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
            v19[_iv0] = _vec[_iv0];
          }
        }	// L26
        float y[2];	// L27
        for (int v21 = 0; v21 < 2; v21++) {	// L28
          y[v21] = (float)0.000000;	// L28
        }
        ap_int<33> v22 = h;	// L29
        bool v23 = v22 == 0;	// L30
        if (v23) {	// L31
          float v24 = ar[c];	// L32
          float v25 = br[c];	// L33
          float v26 = v24 - v25;	// L34
          float dr;	// L35
          dr = v26;	// L36
          float v28 = ai[c];	// L37
          float v29 = bi[c];	// L38
          float v30 = v28 - v29;	// L39
          float di;	// L40
          di = v30;	// L41
          int32_t v32 = ell;	// L42
          int v33 = v32;	// L43
          float v34 = _st_tw[c][v33][0];	// L44
          float wr;	// L45
          wr = v34;	// L46
          int32_t v36 = ell;	// L47
          int v37 = v36;	// L48
          float v38 = _st_tw[c][v37][1];	// L49
          float wi;	// L50
          wi = v38;	// L51
          float v40 = dr;	// L52
          float v41 = wr;	// L53
          float v42 = v40 * v41;	// L54
          float v43 = di;	// L55
          float v44 = wi;	// L56
          float v45 = v43 * v44;	// L57
          float v46 = v42 - v45;	// L58
          y[0] = v46;	// L59
          float v47 = dr;	// L60
          float v48 = wi;	// L61
          float v49 = v47 * v48;	// L62
          float v50 = di;	// L63
          float v51 = wr;	// L64
          float v52 = v50 * v51;	// L65
          float v53 = v49 + v52;	// L66
          y[1] = v53;	// L67
          float v54 = v19[0];	// L68
          ar[c] = v54;	// L69
          float v55 = v19[1];	// L70
          ai[c] = v55;	// L71
        } else {
          float v56 = ar[c];	// L73
          float v57 = v19[0];	// L74
          float v58 = v56 + v57;	// L75
          y[0] = v58;	// L76
          float v59 = ai[c];	// L77
          float v60 = v19[1];	// L78
          float v61 = v59 + v60;	// L79
          y[1] = v61;	// L80
          float v62 = v19[0];	// L81
          br[c] = v62;	// L82
          float v63 = v19[1];	// L83
          bi[c] = v63;	// L84
        }
        {
          hls::vector< float, 2 > _vec;
          for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
            _vec[_iv0] = y[_iv0];
          }
          v2.write(_vec);
        }	// L86
      }
    }
  }
}

/// This is top function.
void top(

) {	// L92
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v64;
  #pragma HLS stream variable=v64 depth=2	// L93
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v65;
  #pragma HLS stream variable=v65 depth=2	// L94
  hls::stream< int32_t > v66;
  #pragma HLS stream variable=v66 depth=1	// L95
  delay6_r0_0(v66, v64, v65);	// L96
}

