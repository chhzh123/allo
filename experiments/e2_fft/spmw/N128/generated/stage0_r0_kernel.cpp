
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
float _st_tw[64][2] = {1.000000e+00, -0.000000e+00, 9.987954e-01, -4.906768e-02, 9.951847e-01, -9.801714e-02, 9.891765e-01, -1.467305e-01, 9.807853e-01, -1.950903e-01, 9.700313e-01, -2.429802e-01, 9.569404e-01, -2.902847e-01, 9.415441e-01, -3.368899e-01, 9.238795e-01, -3.826834e-01, 9.039893e-01, -4.275551e-01, 8.819213e-01, -4.713967e-01, 8.577286e-01, -5.141028e-01, 8.314696e-01, -5.555702e-01, 8.032075e-01, -5.956993e-01, 7.730104e-01, -6.343933e-01, 7.409511e-01, -6.715590e-01, 7.071068e-01, -7.071068e-01, 6.715590e-01, -7.409511e-01, 6.343933e-01, -7.730104e-01, 5.956993e-01, -8.032075e-01, 5.555702e-01, -8.314696e-01, 5.141028e-01, -8.577286e-01, 4.713967e-01, -8.819213e-01, 4.275551e-01, -9.039893e-01, 3.826834e-01, -9.238795e-01, 3.368899e-01, -9.415441e-01, 2.902847e-01, -9.569404e-01, 2.429802e-01, -9.700313e-01, 1.950903e-01, -9.807853e-01, 1.467305e-01, -9.891765e-01, 9.801714e-02, -9.951847e-01, 4.906768e-02, -9.987954e-01, 6.123234e-17, -1.000000e+00, -4.906768e-02, -9.987954e-01, -9.801714e-02, -9.951847e-01, -1.467305e-01, -9.891765e-01, -1.950903e-01, -9.807853e-01, -2.429802e-01, -9.700313e-01, -2.902847e-01, -9.569404e-01, -3.368899e-01, -9.415441e-01, -3.826834e-01, -9.238795e-01, -4.275551e-01, -9.039893e-01, -4.713967e-01, -8.819213e-01, -5.141028e-01, -8.577286e-01, -5.555702e-01, -8.314696e-01, -5.956993e-01, -8.032075e-01, -6.343933e-01, -7.730104e-01, -6.715590e-01, -7.409511e-01, -7.071068e-01, -7.071068e-01, -7.409511e-01, -6.715590e-01, -7.730104e-01, -6.343933e-01, -8.032075e-01, -5.956993e-01, -8.314696e-01, -5.555702e-01, -8.577286e-01, -5.141028e-01, -8.819213e-01, -4.713967e-01, -9.039893e-01, -4.275551e-01, -9.238795e-01, -3.826834e-01, -9.415441e-01, -3.368899e-01, -9.569404e-01, -2.902847e-01, -9.700313e-01, -2.429802e-01, -9.807853e-01, -1.950903e-01, -9.891765e-01, -1.467305e-01, -9.951847e-01, -9.801714e-02, -9.987954e-01, -4.906768e-02};	// L2
void stage0_r0_0(
  hls::stream< hls::vector< float, 2 > >& v0,
  hls::stream< hls::vector< float, 2 > >& v1
) {	// L3
  // placeholder for const float _st_tw	// L8
  float ar[64];	// L9
  for (int v4 = 0; v4 < 64; v4++) {	// L10
    ar[v4] = (float)0.000000;	// L10
  }
  float ai[64];	// L11
  for (int v6 = 0; v6 < 64; v6++) {	// L12
    ai[v6] = (float)0.000000;	// L12
  }
  float br[64];	// L13
  for (int v8 = 0; v8 < 64; v8++) {	// L14
    br[v8] = (float)0.000000;	// L14
  }
  float bi[64];	// L15
  for (int v10 = 0; v10 < 64; v10++) {	// L16
    bi[v10] = (float)0.000000;	// L16
  }
  l_S__b_0__b: for (int _b = 0; _b < 34; _b++) {	// L17
    l_S_h_0_h: for (int h = 0; h < 2; h++) {	// L18
      l_S_c_0_c: for (int c = 0; c < 64; c++) {	// L19
        float v14[2];
        {
          hls::vector< float, 2 > _vec = v0.read();
          for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
            v14[_iv0] = _vec[_iv0];
          }
        }	// L20
        float y[2];	// L21
        for (int v16 = 0; v16 < 2; v16++) {	// L22
          y[v16] = (float)0.000000;	// L22
        }
        ap_int<33> v17 = h;	// L23
        bool v18 = v17 == 0;	// L24
        if (v18) {	// L25
          float v19 = ar[c];	// L26
          float v20 = br[c];	// L27
          float v21 = v19 - v20;	// L28
          float dr;	// L29
          dr = v21;	// L30
          float v23 = ai[c];	// L31
          float v24 = bi[c];	// L32
          float v25 = v23 - v24;	// L33
          float di;	// L34
          di = v25;	// L35
          int32_t v27 = c;	// L36
          int32_t k;	// L37
          k = v27;	// L38
          int32_t v29 = k;	// L39
          int v30 = v29;	// L40
          float v31 = _st_tw[v30][0];	// L41
          float wr;	// L42
          wr = v31;	// L43
          int32_t v33 = k;	// L44
          int v34 = v33;	// L45
          float v35 = _st_tw[v34][1];	// L46
          float wi;	// L47
          wi = v35;	// L48
          float v37 = dr;	// L49
          float v38 = wr;	// L50
          float v39 = v37 * v38;	// L51
          float v40 = di;	// L52
          float v41 = wi;	// L53
          float v42 = v40 * v41;	// L54
          float v43 = v39 - v42;	// L55
          y[0] = v43;	// L56
          float v44 = dr;	// L57
          float v45 = wi;	// L58
          float v46 = v44 * v45;	// L59
          float v47 = di;	// L60
          float v48 = wr;	// L61
          float v49 = v47 * v48;	// L62
          float v50 = v46 + v49;	// L63
          y[1] = v50;	// L64
          float v51 = v14[0];	// L65
          ar[c] = v51;	// L66
          float v52 = v14[1];	// L67
          ai[c] = v52;	// L68
        } else {
          float v53 = ar[c];	// L70
          float v54 = v14[0];	// L71
          float v55 = v53 + v54;	// L72
          y[0] = v55;	// L73
          float v56 = ai[c];	// L74
          float v57 = v14[1];	// L75
          float v58 = v56 + v57;	// L76
          y[1] = v58;	// L77
          float v59 = v14[0];	// L78
          br[c] = v59;	// L79
          float v60 = v14[1];	// L80
          bi[c] = v60;	// L81
        }
        {
          hls::vector< float, 2 > _vec;
          for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
            _vec[_iv0] = y[_iv0];
          }
          v1.write(_vec);
        }	// L83
      }
    }
  }
}

/// This is top function.
void top(

) {	// L89
  #pragma HLS dataflow
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v61;
  #pragma HLS stream variable=v61 depth=2	// L90
  // Stream of vectors: each vector packs float array[2] into hls::vector<float, 2>
  hls::stream< hls::vector< float, 2 > > v62;
  #pragma HLS stream variable=v62 depth=2	// L91
  stage0_r0_0(v61, v62);	// L92
}

