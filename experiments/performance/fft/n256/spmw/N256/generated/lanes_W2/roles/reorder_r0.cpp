
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
int32_t _st_rd[128] = {0, 32, 16, 48, 8, 40, 24, 56, 4, 36, 20, 52, 12, 44, 28, 60, 2, 34, 18, 50, 10, 42, 26, 58, 6, 38, 22, 54, 14, 46, 30, 62, 1, 33, 17, 49, 9, 41, 25, 57, 5, 37, 21, 53, 13, 45, 29, 61, 3, 35, 19, 51, 11, 43, 27, 59, 7, 39, 23, 55, 15, 47, 31, 63, 64, 96, 80, 112, 72, 104, 88, 120, 68, 100, 84, 116, 76, 108, 92, 124, 66, 98, 82, 114, 74, 106, 90, 122, 70, 102, 86, 118, 78, 110, 94, 126, 65, 97, 81, 113, 73, 105, 89, 121, 69, 101, 85, 117, 77, 109, 93, 125, 67, 99, 83, 115, 75, 107, 91, 123, 71, 103, 87, 119, 79, 111, 95, 127};	// L2
void reorder_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1,
  hls::stream< hls::vector< float, 2 > >& v2,
  hls::stream< hls::vector< float, 2 > >& v3
) {	// L3
  // placeholder for const int32_t _st_rd	// L8
  float ar[2][128];	// L9
  for (int v6 = 0; v6 < 2; v6++) {	// L10
    for (int v7 = 0; v7 < 128; v7++) {	// L10
      ar[v6][v7] = (float)0.000000;	// L10
    }
  }
  float ai[2][128];	// L11
  for (int v9 = 0; v9 < 2; v9++) {	// L12
    for (int v10 = 0; v10 < 128; v10++) {	// L12
      ai[v9][v10] = (float)0.000000;	// L12
    }
  }
  float br[2][128];	// L13
  for (int v12 = 0; v12 < 2; v12++) {	// L14
    for (int v13 = 0; v13 < 128; v13++) {	// L14
      br[v12][v13] = (float)0.000000;	// L14
    }
  }
  float bi[2][128];	// L15
  for (int v15 = 0; v15 < 2; v15++) {	// L16
    for (int v16 = 0; v16 < 128; v16++) {	// L16
      bi[v15][v16] = (float)0.000000;	// L16
    }
  }
  l_S__f_0__f: for (int _f = 0; _f < 191; _f++) {	// L17
    float v18[2];
    {
      hls::vector< float, 2 > _vec = v0.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v18[_iv0] = _vec[_iv0];
      }
    }	// L18
    float v19[2];
    {
      hls::vector< float, 2 > _vec = v1.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v19[_iv0] = _vec[_iv0];
      }
    }	// L19
  }
  l_S_blk_1_blk: for (int blk = 0; blk < 32; blk++) {	// L21
    int32_t v21 = blk;	// L22
    int32_t v22 = v21 & 1;	// L23
    int32_t side;	// L24
    side = v22;	// L25
    int32_t v24 = side;	// L26
    ap_int<33> v25 = v24;	// L27
    ap_int<33> v26 = 1 - v25;	// L28
    int32_t v27 = v26;	// L29
    int32_t other;	// L30
    other = v27;	// L31
    l_S_r_1_r: for (int r = 0; r < 128; r++) {	// L32
      float v30[2];
      {
        hls::vector< float, 2 > _vec = v0.read();
        for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
          v30[_iv0] = _vec[_iv0];
        }
      }	// L33
      float v31[2];
      {
        hls::vector< float, 2 > _vec = v1.read();
        for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
          v31[_iv0] = _vec[_iv0];
        }
      }	// L34
      int32_t v32 = _st_rd[r];	// L35
      int32_t pos;	// L36
      pos = v32;	// L37
      float v34 = v30[0];	// L38
      int32_t v35 = side;	// L39
      int v36 = v35;	// L40
      int32_t v37 = pos;	// L41
      int v38 = v37;	// L42
      ar[v36][v38] = v34;	// L43
      float v39 = v30[1];	// L44
      int32_t v40 = side;	// L45
      int v41 = v40;	// L46
      int32_t v42 = pos;	// L47
      int v43 = v42;	// L48
      ai[v41][v43] = v39;	// L49
      float v44 = v31[0];	// L50
      int32_t v45 = side;	// L51
      int v46 = v45;	// L52
      int32_t v47 = pos;	// L53
      int v48 = v47;	// L54
      br[v46][v48] = v44;	// L55
      float v49 = v31[1];	// L56
      int32_t v50 = side;	// L57
      int v51 = v50;	// L58
      int32_t v52 = pos;	// L59
      int v53 = v52;	// L60
      bi[v51][v53] = v49;	// L61
      ap_int<33> v54 = blk;	// L62
      bool v55 = v54 > 0;	// L63
      if (v55) {	// L64
        float lo[2];	// L65
        for (int v57 = 0; v57 < 2; v57++) {	// L66
          lo[v57] = (float)0.000000;	// L66
        }
        float hi[2];	// L67
        for (int v59 = 0; v59 < 2; v59++) {	// L68
          hi[v59] = (float)0.000000;	// L68
        }
        int32_t v60 = other;	// L69
        int v61 = v60;	// L70
        float v62 = ar[v61][r];	// L71
        lo[0] = v62;	// L72
        int32_t v63 = other;	// L73
        int v64 = v63;	// L74
        float v65 = ai[v64][r];	// L75
        lo[1] = v65;	// L76
        int32_t v66 = other;	// L77
        int v67 = v66;	// L78
        float v68 = br[v67][r];	// L79
        hi[0] = v68;	// L80
        int32_t v69 = other;	// L81
        int v70 = v69;	// L82
        float v71 = bi[v70][r];	// L83
        hi[1] = v71;	// L84
        {
          hls::vector< float, 2 > _vec;
          for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
            _vec[_iv0] = lo[_iv0];
          }
          v2.write(_vec);
        }	// L85
        {
          hls::vector< float, 2 > _vec;
          for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
            _vec[_iv0] = hi[_iv0];
          }
          v3.write(_vec);
        }	// L86
      }
    }
  }
  int32_t last;	// L90
  last = 1;	// L91
  l_S_r2_3_r2: for (int r2 = 0; r2 < 128; r2++) {	// L92
    float tlo[2];	// L93
    for (int v75 = 0; v75 < 2; v75++) {	// L94
      tlo[v75] = (float)0.000000;	// L94
    }
    float thi[2];	// L95
    for (int v77 = 0; v77 < 2; v77++) {	// L96
      thi[v77] = (float)0.000000;	// L96
    }
    int32_t v78 = last;	// L97
    int v79 = v78;	// L98
    float v80 = ar[v79][r2];	// L99
    tlo[0] = v80;	// L100
    int32_t v81 = last;	// L101
    int v82 = v81;	// L102
    float v83 = ai[v82][r2];	// L103
    tlo[1] = v83;	// L104
    int32_t v84 = last;	// L105
    int v85 = v84;	// L106
    float v86 = br[v85][r2];	// L107
    thi[0] = v86;	// L108
    int32_t v87 = last;	// L109
    int v88 = v87;	// L110
    float v89 = bi[v88][r2];	// L111
    thi[1] = v89;	// L112
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = tlo[_iv0];
      }
      v2.write(_vec);
    }	// L113
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = thi[_iv0];
      }
      v3.write(_vec);
    }	// L114
  }
}

/// This is top function.
void top(

) {	// L118
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v90;
  #pragma HLS stream variable=v90 depth=8	// L119
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v91;
  #pragma HLS stream variable=v91 depth=8	// L120
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v92;
  #pragma HLS stream variable=v92 depth=8	// L121
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v93;
  #pragma HLS stream variable=v93 depth=8	// L122
  reorder_r0_0(v90, v92, v91, v93);	// L123
}

