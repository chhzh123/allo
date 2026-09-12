
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
int32_t _st_sel[128][1] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};	// L2
void bfly7_r0_0(
  hls::stream< int32_t >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3,
  hls::stream< hls::vector< float, 2 > >& v4
) {	// L3
  // placeholder for const int32_t _st_sel	// L7
  int32_t v6 = v0.read();	// L8
  int32_t _st__pid0;	// L9
  _st__pid0 = v6;	// L10
  int32_t v8 = _st__pid0;	// L11
  int32_t m;	// L12
  m = v8;	// L13
  l_S_t_0_t: for (int t = 0; t < 4287; t++) {	// L14
    float v11[2];
    {
      hls::vector< float, 2 > _vec = v1.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v11[_iv0] = _vec[_iv0];
      }
    }	// L15
    float v12[2];
    {
      hls::vector< float, 2 > _vec = v2.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v12[_iv0] = _vec[_iv0];
      }
    }	// L16
    ap_int<33> v13 = t;	// L17
    ap_int<33> v14 = v13 & 127;	// L18
    int32_t v15 = v14;	// L19
    int32_t r;	// L20
    r = v15;	// L21
    float v17 = v11[0];	// L22
    float v18 = v12[0];	// L23
    float v19 = v17 - v18;	// L24
    #pragma HLS bind_op variable=v19 op=fsub impl=fabric
    float dr;	// L25
    dr = v19;	// L26
    float v21 = v11[1];	// L27
    float v22 = v12[1];	// L28
    float v23 = v21 - v22;	// L29
    #pragma HLS bind_op variable=v23 op=fsub impl=fabric
    float di;	// L30
    di = v23;	// L31
    int32_t v25 = r;	// L32
    int v26 = v25;	// L33
    int32_t v27 = m;	// L34
    int v28 = v27;	// L35
    int32_t v29 = _st_sel[v26][v28];	// L36
    int32_t rot;	// L37
    rot = v29;	// L38
    float p[2];	// L39
    for (int v32 = 0; v32 < 2; v32++) {	// L40
      p[v32] = (float)0.000000;	// L40
    }
    float q[2];	// L41
    for (int v34 = 0; v34 < 2; v34++) {	// L42
      q[v34] = (float)0.000000;	// L42
    }
    float v35 = v11[0];	// L43
    float v36 = v12[0];	// L44
    float v37 = v35 + v36;	// L45
    #pragma HLS bind_op variable=v37 op=fadd impl=fabric
    p[0] = v37;	// L46
    float v38 = v11[1];	// L47
    float v39 = v12[1];	// L48
    float v40 = v38 + v39;	// L49
    #pragma HLS bind_op variable=v40 op=fadd impl=fabric
    p[1] = v40;	// L50
    float v41 = dr;	// L51
    q[0] = v41;	// L52
    float v42 = di;	// L53
    q[1] = v42;	// L54
    int32_t v43 = rot;	// L55
    bool v44 = v43 == 1;	// L56
    if (v44) {	// L57
      float v45 = di;	// L58
      q[0] = v45;	// L59
      float v46 = dr;	// L60
      float v47 = (float)0.000000 - v46;	// L61
      #pragma HLS bind_op variable=v47 op=fsub impl=fabric
      q[1] = v47;	// L62
    }
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = p[_iv0];
      }
      v3.write(_vec);
    }	// L64
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = q[_iv0];
      }
      v4.write(_vec);
    }	// L65
  }
}

/// This is top function.
void top(

) {	// L69
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v48;
  #pragma HLS stream variable=v48 depth=8	// L70
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v49;
  #pragma HLS stream variable=v49 depth=8	// L71
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v50;
  #pragma HLS stream variable=v50 depth=8	// L72
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v51;
  #pragma HLS stream variable=v51 depth=8	// L73
  hls::stream< int32_t > v52;
  #pragma HLS stream variable=v52 depth=1	// L74
  bfly7_r0_0(v52, v48, v50, v49, v51);	// L75
}

