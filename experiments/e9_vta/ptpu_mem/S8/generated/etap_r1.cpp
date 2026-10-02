
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
void etap_r1_0(
  hls::stream< int16_t >& v0,
  hls::stream< hls::vector< int8_t, 17 > >& v1,
  hls::stream< int8_t >& v2
) {	// L2
  bool go;	// L6
  go = 1;	// L7
  while (true) {	// L8
    #pragma HLS pipeline II=1
    #pragma HLS latency max=0
    bool v4 = go;	// L9
    if (!(v4)) break;
    int8_t v5[17];
    {
      hls::vector< int8_t, 17 > _vec = v1.read();
      for (int _iv0 = 0; _iv0 < 17; ++_iv0) {
        v5[_iv0] = _vec[_iv0];
      }
    }	// L16
    int8_t v6 = v5[16];	// L17
    int32_t v7 = v6;	// L18
    int32_t v8 = v7 & 3;	// L21
    int16_t v9 = v8;	// L22
    int16_t f;	// L23
    f = v9;	// L24
    int8_t v11 = v5[0];	// L25
    int32_t v12 = v11;	// L26
    int32_t v13 = v12 & 255;	// L29
    int16_t v14 = v13;	// L30
    int16_t a;	// L31
    a = v14;	// L32
    int16_t v16 = a;	// L33
    int16_t v17 = f;	// L34
    int16_t v18 = v17 << 8;	// L38
    int16_t v19 = v16 | v18;	// L39
    v0.write(v19);	// L40
    int8_t v20 = v5[8];	// L41
    v2.write(v20);	// L42
    l_S_k_0_k: for (int k = 0; k < 7; k++) {	// L43
      int8_t v22 = v5[(k + 1)];	// L44
      v5[k] = v22;	// L45
      int8_t v23 = v5[(k + 9)];	// L46
      v5[(k + 8)] = v23;	// L47
    }
    v5[7] = 0;	// L52
    v5[15] = 0;	// L56
    int16_t v24 = f;	// L57
    int16_t v25 = v24 >> 1;	// L61
    int32_t v26 = v25;	// L62
    int32_t v27 = v26 & 1;	// L65
    bool v28 = v27 != 0;	// L68
    if (v28) {	// L69
      go = 0;	// L73
    }
  }
}

