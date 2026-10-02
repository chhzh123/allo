
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void etap_r2_0(
  hls::stream< int16_t >& v0,
  hls::stream< hls::vector< int8_t, 9 > >& v1,
  hls::stream< hls::vector< int8_t, 9 > >& v2,
  hls::stream< int8_t >& v3
) {	// L2
  bool go;	// L6
  go = 1;	// L7
  while (true) {	// L8
    #pragma HLS pipeline II=1
    #pragma HLS latency max=0
    bool v5 = go;	// L9
    if (!(v5)) break;
    int8_t v6[9];
    {
      hls::vector< int8_t, 9 > _vec = v1.read();
      for (int _iv0 = 0; _iv0 < 9; ++_iv0) {
        v6[_iv0] = _vec[_iv0];
      }
    }	// L16
    int8_t v7 = v6[8];	// L17
    int32_t v8 = v7;	// L18
    int32_t v9 = v8 & 3;	// L21
    int16_t v10 = v9;	// L22
    int16_t f;	// L23
    f = v10;	// L24
    int8_t v12 = v6[0];	// L25
    int32_t v13 = v12;	// L26
    int32_t v14 = v13 & 255;	// L29
    int16_t v15 = v14;	// L30
    int16_t a;	// L31
    a = v15;	// L32
    int16_t v17 = a;	// L33
    int16_t v18 = f;	// L34
    int16_t v19 = v18 << 8;	// L38
    int16_t v20 = v17 | v19;	// L39
    v0.write(v20);	// L40
    int8_t v21 = v6[4];	// L41
    v3.write(v21);	// L42
    l_S_k_0_k: for (int k = 0; k < 3; k++) {	// L43
      int8_t v23 = v6[(k + 1)];	// L44
      v6[k] = v23;	// L45
      int8_t v24 = v6[(k + 5)];	// L46
      v6[(k + 4)] = v24;	// L47
    }
    v6[3] = 0;	// L52
    v6[7] = 0;	// L56
    {
      hls::vector< int8_t, 9 > _vec;
      for (int _iv0 = 0; _iv0 < 9; ++_iv0) {
        _vec[_iv0] = v6[_iv0];
      }
      v2.write(_vec);
    }	// L57
    int16_t v25 = f;	// L58
    int16_t v26 = v25 >> 1;	// L62
    int32_t v27 = v26;	// L63
    int32_t v28 = v27 & 1;	// L66
    bool v29 = v28 != 0;	// L69
    if (v29) {	// L70
      go = 0;	// L74
    }
  }
}

