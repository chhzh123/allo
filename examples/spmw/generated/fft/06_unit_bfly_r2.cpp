
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>
using namespace std;
float _st_tw[4][2] = {1.000000e+00, -0.000000e+00, 7.071068e-01, -7.071068e-01, 6.123234e-17, -1.000000e+00, -7.071068e-01, -7.071068e-01};	// L2
void bfly_r2_0(
  hls::stream< int32_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3,
  hls::stream< hls::vector< float, 2 > >& v4,
  hls::stream< hls::vector< float, 2 > >& v5
) {	// L3
  // placeholder for const float _st_tw	// L9
  int32_t v7 = v0.read();	// L10
  int32_t _st__pid0;	// L11
  _st__pid0 = v7;	// L12
  int32_t v9 = v1.read();	// L13
  int32_t _st__pid1;	// L14
  _st__pid1 = v9;	// L15
  int32_t v11 = _st__pid0;	// L16
  int32_t s;	// L17
  s = v11;	// L18
  int32_t v13 = _st__pid1;	// L19
  int32_t b;	// L20
  b = v13;	// L21
  int32_t v15 = s;	// L22
  int32_t v16 = 1 << v15;	// L23
  int32_t span;	// L24
  span = v16;	// L25
  int32_t v18 = b;	// L26
  int32_t v19 = span;	// L27
  int32_t v20 = v18 % v19;	// L28
  int32_t v21 = 4 / v19;	// L30
  int64_t v22 = v20;	// L31
  int64_t v23 = v21;	// L32
  int64_t v24 = v22 * v23;	// L33
  int64_t k;	// L34
  k = v24;	// L35
  int64_t v26 = k;	// L36
  int v27 = v26;	// L37
  float v28 = _st_tw[v27][0];	// L38
  float wr;	// L39
  wr = v28;	// L40
  int64_t v30 = k;	// L41
  int v31 = v30;	// L42
  float v32 = _st_tw[v31][1];	// L43
  float wi;	// L44
  wi = v32;	// L45
  float v34[2];
  {
    hls::vector< float, 2 > _vec = v2.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v34[_iv0] = _vec[_iv0];
    }
  }	// L46
  float v35[2];
  {
    hls::vector< float, 2 > _vec = v3.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v35[_iv0] = _vec[_iv0];
    }
  }	// L47
  float v36 = wr;	// L48
  float v37 = v35[0];	// L49
  float v38 = v36 * v37;	// L50
  float v39 = wi;	// L51
  float v40 = v35[1];	// L52
  float v41 = v39 * v40;	// L53
  float v42 = v38 - v41;	// L54
  float tr;	// L55
  tr = v42;	// L56
  float v44 = wr;	// L57
  float v45 = v35[1];	// L58
  float v46 = v44 * v45;	// L59
  float v47 = wi;	// L60
  float v48 = v35[0];	// L61
  float v49 = v47 * v48;	// L62
  float v50 = v46 + v49;	// L63
  float ti;	// L64
  ti = v50;	// L65
  float u[2];	// L66
  for (int v53 = 0; v53 < 2; v53++) {	// L67
    u[v53] = (float)0.000000;	// L67
  }
  float l[2];	// L68
  for (int v55 = 0; v55 < 2; v55++) {	// L69
    l[v55] = (float)0.000000;	// L69
  }
  float v56 = v34[0];	// L70
  float v57 = tr;	// L71
  float v58 = v56 + v57;	// L72
  u[0] = v58;	// L73
  float v59 = v34[1];	// L74
  float v60 = ti;	// L75
  float v61 = v59 + v60;	// L76
  u[1] = v61;	// L77
  float v62 = v34[0];	// L78
  float v63 = tr;	// L79
  float v64 = v62 - v63;	// L80
  l[0] = v64;	// L81
  float v65 = v34[1];	// L82
  float v66 = ti;	// L83
  float v67 = v65 - v66;	// L84
  l[1] = v67;	// L85
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u[_iv0];
    }
    v4.write(_vec);
  }	// L86
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l[_iv0];
    }
    v5.write(_vec);
  }	// L87
}

/// This is top function.
void top(

) {	// L90
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v68;
  #pragma HLS stream variable=v68 depth=2	// L91
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v69;
  #pragma HLS stream variable=v69 depth=2	// L92
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v70;
  #pragma HLS stream variable=v70 depth=2	// L93
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v71;
  #pragma HLS stream variable=v71 depth=2	// L94
  hls::stream< int32_t > v72;
  #pragma HLS stream variable=v72 depth=1	// L95
  hls::stream< int32_t > v73;
  #pragma HLS stream variable=v73 depth=1	// L96
  bfly_r2_0(v72, v73, v70, v68, v71, v69);	// L97
}

