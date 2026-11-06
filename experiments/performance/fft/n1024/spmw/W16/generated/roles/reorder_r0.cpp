
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
int32_t _st_rd[64] = {0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31, 32, 48, 40, 56, 36, 52, 44, 60, 34, 50, 42, 58, 38, 54, 46, 62, 33, 49, 41, 57, 37, 53, 45, 61, 35, 51, 43, 59, 39, 55, 47, 63};	// L2
/// This is top function.
void reorder_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3
) {	// L3
  // placeholder for const int32_t _st_rd	// L4
  float ar[2][64];	// L8
  for (int v6 = 0; v6 < 2; v6++) {	// L9
    for (int v7 = 0; v7 < 64; v7++) {	// L9
      ar[v6][v7] = (float)0.000000;	// L9
    }
  }
  float ai[2][64];	// L13
  for (int v9 = 0; v9 < 2; v9++) {	// L14
    for (int v10 = 0; v10 < 64; v10++) {	// L14
      ai[v9][v10] = (float)0.000000;	// L14
    }
  }
  float br[2][64];	// L18
  for (int v12 = 0; v12 < 2; v12++) {	// L19
    for (int v13 = 0; v13 < 64; v13++) {	// L19
      br[v12][v13] = (float)0.000000;	// L19
    }
  }
  float bi[2][64];	// L23
  for (int v15 = 0; v15 < 2; v15++) {	// L24
    for (int v16 = 0; v16 < 64; v16++) {	// L24
      bi[v15][v16] = (float)0.000000;	// L24
    }
  }
  l_S__f_0__f: for (int _f = 0; _f < 95; _f++) {	// L25
    float v18[2];
    {
      hls::vector< float, 2 > _vec = v0.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v18[_iv0] = _vec[_iv0];
      }
    }	// L26
    float v19[2];
    {
      hls::vector< float, 2 > _vec = v2.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v19[_iv0] = _vec[_iv0];
      }
    }	// L27
  }
  l_S_blk_1_blk: for (int blk = 0; blk < 32; blk++) {	// L29
    int32_t v21 = blk;	// L30
    int32_t v22 = v21 & 1;	// L33
    int32_t side;	// L34
    side = v22;	// L35
    int32_t v24 = side;	// L36
    ap_int<33> v25 = v24;	// L40
    ap_int<33> v26 = 1 - v25;	// L41
    int32_t v27 = v26;	// L42
    int32_t other;	// L43
    other = v27;	// L44
    l_S_r_1_r: for (int r = 0; r < 64; r++) {	// L45
      float v30[2];
      {
        hls::vector< float, 2 > _vec = v0.read();
        for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
          v30[_iv0] = _vec[_iv0];
        }
      }	// L46
      float v31[2];
      {
        hls::vector< float, 2 > _vec = v2.read();
        for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
          v31[_iv0] = _vec[_iv0];
        }
      }	// L47
      int32_t v32 = _st_rd[r];	// L48
      int32_t pos;	// L49
      pos = v32;	// L50
      float v34 = v30[0];	// L51
      int32_t v35 = side;	// L52
      int v36 = v35;	// L53
      int32_t v37 = pos;	// L54
      int v38 = v37;	// L55
      ar[v36][v38] = v34;	// L56
      float v39 = v30[1];	// L57
      int32_t v40 = side;	// L58
      int v41 = v40;	// L59
      int32_t v42 = pos;	// L60
      int v43 = v42;	// L61
      ai[v41][v43] = v39;	// L62
      float v44 = v31[0];	// L63
      int32_t v45 = side;	// L64
      int v46 = v45;	// L65
      int32_t v47 = pos;	// L66
      int v48 = v47;	// L67
      br[v46][v48] = v44;	// L68
      float v49 = v31[1];	// L69
      int32_t v50 = side;	// L70
      int v51 = v50;	// L71
      int32_t v52 = pos;	// L72
      int v53 = v52;	// L73
      bi[v51][v53] = v49;	// L74
      ap_int<33> v54 = blk;	// L75
      bool v55 = v54 > 0;	// L79
      if (v55) {	// L80
        float lo[2];	// L84
        for (int v57 = 0; v57 < 2; v57++) {	// L85
          lo[v57] = (float)0.000000;	// L85
        }
        float hi[2];	// L89
        for (int v59 = 0; v59 < 2; v59++) {	// L90
          hi[v59] = (float)0.000000;	// L90
        }
        int32_t v60 = other;	// L91
        int v61 = v60;	// L92
        float v62 = ar[v61][r];	// L93
        lo[0] = v62;	// L94
        int32_t v63 = other;	// L95
        int v64 = v63;	// L96
        float v65 = ai[v64][r];	// L97
        lo[1] = v65;	// L98
        int32_t v66 = other;	// L99
        int v67 = v66;	// L100
        float v68 = br[v67][r];	// L101
        hi[0] = v68;	// L102
        int32_t v69 = other;	// L103
        int v70 = v69;	// L104
        float v71 = bi[v70][r];	// L105
        hi[1] = v71;	// L106
        {
          hls::vector< float, 2 > _vec;
          for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
            _vec[_iv0] = lo[_iv0];
          }
          v1.write(_vec);
        }	// L107
        {
          hls::vector< float, 2 > _vec;
          for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
            _vec[_iv0] = hi[_iv0];
          }
          v3.write(_vec);
        }	// L108
      }
    }
  }
  int32_t last;	// L124
  last = 1;	// L125
  l_S_r2_3_r2: for (int r2 = 0; r2 < 64; r2++) {	// L126
    float tlo[2];	// L130
    for (int v75 = 0; v75 < 2; v75++) {	// L131
      tlo[v75] = (float)0.000000;	// L131
    }
    float thi[2];	// L135
    for (int v77 = 0; v77 < 2; v77++) {	// L136
      thi[v77] = (float)0.000000;	// L136
    }
    int32_t v78 = last;	// L137
    int v79 = v78;	// L138
    float v80 = ar[v79][r2];	// L139
    tlo[0] = v80;	// L140
    int32_t v81 = last;	// L141
    int v82 = v81;	// L142
    float v83 = ai[v82][r2];	// L143
    tlo[1] = v83;	// L144
    int32_t v84 = last;	// L145
    int v85 = v84;	// L146
    float v86 = br[v85][r2];	// L147
    thi[0] = v86;	// L148
    int32_t v87 = last;	// L149
    int v88 = v87;	// L150
    float v89 = bi[v88][r2];	// L151
    thi[1] = v89;	// L152
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = tlo[_iv0];
      }
      v1.write(_vec);
    }	// L153
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = thi[_iv0];
      }
      v3.write(_vec);
    }	// L154
  }
}

