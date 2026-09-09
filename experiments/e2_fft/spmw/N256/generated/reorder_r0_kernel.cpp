
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
int32_t _st_perm[256] = {0, 128, 64, 192, 32, 160, 96, 224, 16, 144, 80, 208, 48, 176, 112, 240, 8, 136, 72, 200, 40, 168, 104, 232, 24, 152, 88, 216, 56, 184, 120, 248, 4, 132, 68, 196, 36, 164, 100, 228, 20, 148, 84, 212, 52, 180, 116, 244, 12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124, 252, 2, 130, 66, 194, 34, 162, 98, 226, 18, 146, 82, 210, 50, 178, 114, 242, 10, 138, 74, 202, 42, 170, 106, 234, 26, 154, 90, 218, 58, 186, 122, 250, 6, 134, 70, 198, 38, 166, 102, 230, 22, 150, 86, 214, 54, 182, 118, 246, 14, 142, 78, 206, 46, 174, 110, 238, 30, 158, 94, 222, 62, 190, 126, 254, 1, 129, 65, 193, 33, 161, 97, 225, 17, 145, 81, 209, 49, 177, 113, 241, 9, 137, 73, 201, 41, 169, 105, 233, 25, 153, 89, 217, 57, 185, 121, 249, 5, 133, 69, 197, 37, 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245, 13, 141, 77, 205, 45, 173, 109, 237, 29, 157, 93, 221, 61, 189, 125, 253, 3, 131, 67, 195, 35, 163, 99, 227, 19, 147, 83, 211, 51, 179, 115, 243, 11, 139, 75, 203, 43, 171, 107, 235, 27, 155, 91, 219, 59, 187, 123, 251, 7, 135, 71, 199, 39, 167, 103, 231, 23, 151, 87, 215, 55, 183, 119, 247, 15, 143, 79, 207, 47, 175, 111, 239, 31, 159, 95, 223, 63, 191, 127, 255};	// L2
void reorder_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1
) {	// L3
  // placeholder for const int32_t _st_perm	// L9
  float bufr[2][256];	// L10
  for (int v4 = 0; v4 < 2; v4++) {	// L11
    for (int v5 = 0; v5 < 256; v5++) {	// L11
      bufr[v4][v5] = (float)0.000000;	// L11
    }
  }
  float bufi[2][256];	// L12
  for (int v7 = 0; v7 < 2; v7++) {	// L13
    for (int v8 = 0; v8 < 256; v8++) {	// L13
      bufi[v7][v8] = (float)0.000000;	// L13
    }
  }
  l_S__t_0__t: for (int _t = 0; _t < 255; _t++) {	// L14
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
    l_S_i_1_i: for (int i = 0; i < 256; i++) {	// L28
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
  l_S_i_3_i1: for (int i1 = 0; i1 < 256; i1++) {	// L64
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

