
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
int32_t _st_sel[512][1] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};	// L2
/// This is top function.
void bfly9_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3,
  hls::stream< int32_t >& v4
) {	// L3
  // placeholder for const int32_t _st_sel	// L4
  int32_t v6 = v4.read();	// L5
  int32_t _st__pid0;	// L6
  _st__pid0 = v6;	// L7
  int32_t v8 = _st__pid0;	// L8
  int32_t m;	// L9
  m = v8;	// L10
  l_S_t_0_t: for (int t = 0; t < 17151; t++) {	// L11
    float v11[2];
    {
      hls::vector< float, 2 > _vec = v0.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v11[_iv0] = _vec[_iv0];
      }
    }	// L12
    float v12[2];
    {
      hls::vector< float, 2 > _vec = v2.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v12[_iv0] = _vec[_iv0];
      }
    }	// L13
    ap_int<33> v13 = t;	// L21
    ap_int<33> v14 = v13 & 511;	// L22
    int32_t v15 = v14;	// L23
    int32_t r;	// L24
    r = v15;	// L25
    float v17 = v11[0];	// L26
    float v18 = v12[0];	// L27
    float v19 = v17 - v18;	// L28
    #pragma HLS bind_op variable=v19 op=fsub impl=fabric
    float dr;	// L29
    dr = v19;	// L30
    float v21 = v11[1];	// L31
    float v22 = v12[1];	// L32
    float v23 = v21 - v22;	// L33
    #pragma HLS bind_op variable=v23 op=fsub impl=fabric
    float di;	// L34
    di = v23;	// L35
    int32_t v25 = r;	// L36
    int v26 = v25;	// L37
    int32_t v27 = m;	// L38
    int v28 = v27;	// L39
    int32_t v29 = _st_sel[v26][v28];	// L40
    int32_t rot;	// L41
    rot = v29;	// L42
    float p[2];	// L46
    for (int v32 = 0; v32 < 2; v32++) {	// L47
      p[v32] = (float)0.000000;	// L47
    }
    float q[2];	// L51
    for (int v34 = 0; v34 < 2; v34++) {	// L52
      q[v34] = (float)0.000000;	// L52
    }
    float v35 = v11[0];	// L53
    float v36 = v12[0];	// L54
    float v37 = v35 + v36;	// L55
    #pragma HLS bind_op variable=v37 op=fadd impl=fabric
    p[0] = v37;	// L56
    float v38 = v11[1];	// L57
    float v39 = v12[1];	// L58
    float v40 = v38 + v39;	// L59
    #pragma HLS bind_op variable=v40 op=fadd impl=fabric
    p[1] = v40;	// L60
    float v41 = dr;	// L61
    q[0] = v41;	// L62
    float v42 = di;	// L63
    q[1] = v42;	// L64
    int32_t v43 = rot;	// L65
    bool v44 = v43 == 1;	// L68
    if (v44) {	// L69
      float v45 = di;	// L70
      q[0] = v45;	// L71
      float v46 = dr;	// L72
      float v47 = (float)0.000000 - v46;	// L75
      #pragma HLS bind_op variable=v47 op=fsub impl=fabric
      q[1] = v47;	// L76
    }
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = p[_iv0];
      }
      v1.write(_vec);
    }	// L78
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = q[_iv0];
      }
      v3.write(_vec);
    }	// L79
  }
}

