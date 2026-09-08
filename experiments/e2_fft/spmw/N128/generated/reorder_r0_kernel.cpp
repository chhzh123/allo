
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
int32_t _st_perm[128] = {0, 64, 32, 96, 16, 80, 48, 112, 8, 72, 40, 104, 24, 88, 56, 120, 4, 68, 36, 100, 20, 84, 52, 116, 12, 76, 44, 108, 28, 92, 60, 124, 2, 66, 34, 98, 18, 82, 50, 114, 10, 74, 42, 106, 26, 90, 58, 122, 6, 70, 38, 102, 22, 86, 54, 118, 14, 78, 46, 110, 30, 94, 62, 126, 1, 65, 33, 97, 17, 81, 49, 113, 9, 73, 41, 105, 25, 89, 57, 121, 5, 69, 37, 101, 21, 85, 53, 117, 13, 77, 45, 109, 29, 93, 61, 125, 3, 67, 35, 99, 19, 83, 51, 115, 11, 75, 43, 107, 27, 91, 59, 123, 7, 71, 39, 103, 23, 87, 55, 119, 15, 79, 47, 111, 31, 95, 63, 127};	// L2
void reorder_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1
) {	// L3
  // placeholder for const int32_t _st_perm	// L9
  float bufr[2][128];	// L10
  for (int v4 = 0; v4 < 2; v4++) {	// L11
    for (int v5 = 0; v5 < 128; v5++) {	// L11
      bufr[v4][v5] = (float)0.000000;	// L11
    }
  }
  float bufi[2][128];	// L12
  for (int v7 = 0; v7 < 2; v7++) {	// L13
    for (int v8 = 0; v8 < 128; v8++) {	// L13
      bufi[v7][v8] = (float)0.000000;	// L13
    }
  }
  l_S__t_0__t: for (int _t = 0; _t < 127; _t++) {	// L14
    float v10[2];
    {
      hls::vector< float, 2 > _vec = v0.read();
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        v10[_iv0] = _vec[_iv0];
      }
    }	// L15
  }
  l_S_b_1_b: for (int b = 0; b < 33; b++) {	// L17
    int32_t v12 = b;	// L18
    int32_t v13 = v12 & 1;	// L19
    int32_t side;	// L20
    side = v13;	// L21
    int32_t v15 = side;	// L22
    ap_int<33> v16 = v15;	// L23
    ap_int<33> v17 = 1 - v16;	// L24
    int32_t v18 = v17;	// L25
    int32_t other;	// L26
    other = v18;	// L27
    l_S_i_1_i: for (int i = 0; i < 128; i++) {	// L28
      float v21[2];
      {
        hls::vector< float, 2 > _vec = v0.read();
        for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
          v21[_iv0] = _vec[_iv0];
        }
      }	// L29
      int32_t v22 = _st_perm[i];	// L30
      int32_t p;	// L31
      p = v22;	// L32
      float v24 = v21[0];	// L33
      int32_t v25 = side;	// L34
      int v26 = v25;	// L35
      int32_t v27 = p;	// L36
      int v28 = v27;	// L37
      bufr[v26][v28] = v24;	// L38
      float v29 = v21[1];	// L39
      int32_t v30 = side;	// L40
      int v31 = v30;	// L41
      int32_t v32 = p;	// L42
      int v33 = v32;	// L43
      bufi[v31][v33] = v29;	// L44
      ap_int<33> v34 = b;	// L45
      bool v35 = v34 > 0;	// L46
      if (v35) {	// L47
        float y[2];	// L48
        for (int v37 = 0; v37 < 2; v37++) {	// L49
          y[v37] = (float)0.000000;	// L49
        }
        int32_t v38 = other;	// L50
        int v39 = v38;	// L51
        float v40 = bufr[v39][i];	// L52
        y[0] = v40;	// L53
        int32_t v41 = other;	// L54
        int v42 = v41;	// L55
        float v43 = bufi[v42][i];	// L56
        y[1] = v43;	// L57
        {
          hls::vector< float, 2 > _vec;
          for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
            _vec[_iv0] = y[_iv0];
          }
          v1.write(_vec);
        }	// L58
      }
    }
  }
  int32_t last;	// L62
  last = 0;	// L63
  l_S_i_3_i1: for (int i1 = 0; i1 < 128; i1++) {	// L64
    float y2[2];	// L65
    for (int v47 = 0; v47 < 2; v47++) {	// L66
      y2[v47] = (float)0.000000;	// L66
    }
    int32_t v48 = last;	// L67
    int v49 = v48;	// L68
    float v50 = bufr[v49][i1];	// L69
    y2[0] = v50;	// L70
    int32_t v51 = last;	// L71
    int v52 = v51;	// L72
    float v53 = bufi[v52][i1];	// L73
    y2[1] = v53;	// L74
    {
      hls::vector< float, 2 > _vec;
      for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
        _vec[_iv0] = y2[_iv0];
      }
      v1.write(_vec);
    }	// L75
  }
}

/// This is top function.
void top(

) {	// L79
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v54;
  #pragma HLS stream variable=v54 depth=2	// L80
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v55;
  #pragma HLS stream variable=v55 depth=2	// L81
  reorder_r0_0(v54, v55);	// L82
}

