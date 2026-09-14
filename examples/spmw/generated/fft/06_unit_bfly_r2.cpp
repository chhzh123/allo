
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
/// This is top function.
void bfly_r2_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3,
  hls::stream< int32_t >& v4,
  hls::stream< int32_t >& v5
) {	// L3
  // placeholder for const float _st_tw	// L4
  int32_t v7 = v4.read();	// L5
  int32_t _st__pid0;	// L6
  _st__pid0 = v7;	// L7
  int32_t v9 = v5.read();	// L8
  int32_t _st__pid1;	// L9
  _st__pid1 = v9;	// L10
  int32_t v11 = _st__pid0;	// L11
  int32_t s;	// L12
  s = v11;	// L13
  int32_t v13 = _st__pid1;	// L14
  int32_t b;	// L15
  b = v13;	// L16
  int32_t v15 = s;	// L17
  int32_t v16 = 1 << v15;	// L20
  int32_t span;	// L21
  span = v16;	// L22
  int32_t v18 = b;	// L23
  int32_t v19 = span;	// L24
  int32_t v20 = v18 % v19;	// L25
  int32_t v21 = 4 / v19;	// L29
  int64_t v22 = v20;	// L30
  int64_t v23 = v21;	// L31
  int64_t v24 = v22 * v23;	// L32
  int64_t k;	// L33
  k = v24;	// L34
  int64_t v26 = k;	// L35
  int v27 = v26;	// L36
  float v28 = _st_tw[v27][0];	// L40
  float wr;	// L41
  wr = v28;	// L42
  int64_t v30 = k;	// L43
  int v31 = v30;	// L44
  float v32 = _st_tw[v31][1];	// L48
  float wi;	// L49
  wi = v32;	// L50
  float v34[2];
  {
    hls::vector< float, 2 > _vec = v2.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v34[_iv0] = _vec[_iv0];
    }
  }	// L51
  float v35[2];
  {
    hls::vector< float, 2 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v35[_iv0] = _vec[_iv0];
    }
  }	// L52
  float v36 = wr;	// L53
  float v37 = v35[0];	// L54
  float v38 = v36 * v37;	// L55
  float v39 = wi;	// L56
  float v40 = v35[1];	// L57
  float v41 = v39 * v40;	// L58
  float v42 = v38 - v41;	// L59
  float tr;	// L60
  tr = v42;	// L61
  float v44 = wr;	// L62
  float v45 = v35[1];	// L63
  float v46 = v44 * v45;	// L64
  float v47 = wi;	// L65
  float v48 = v35[0];	// L66
  float v49 = v47 * v48;	// L67
  float v50 = v46 + v49;	// L68
  float ti;	// L69
  ti = v50;	// L70
  float u[2];	// L74
  for (int v53 = 0; v53 < 2; v53++) {	// L75
    u[v53] = (float)0.000000;	// L75
  }
  float l[2];	// L79
  for (int v55 = 0; v55 < 2; v55++) {	// L80
    l[v55] = (float)0.000000;	// L80
  }
  float v56 = v34[0];	// L81
  float v57 = tr;	// L82
  float v58 = v56 + v57;	// L83
  u[0] = v58;	// L84
  float v59 = v34[1];	// L85
  float v60 = ti;	// L86
  float v61 = v59 + v60;	// L87
  u[1] = v61;	// L88
  float v62 = v34[0];	// L89
  float v63 = tr;	// L90
  float v64 = v62 - v63;	// L91
  l[0] = v64;	// L92
  float v65 = v34[1];	// L93
  float v66 = ti;	// L94
  float v67 = v65 - v66;	// L95
  l[1] = v67;	// L96
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = u[_iv0];
    }
    v3.write(_vec);
  }	// L97
  {
    hls::vector< float, 2 > _vec;
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      _vec[_iv0] = l[_iv0];
    }
    v1.write(_vec);
  }	// L98
}

