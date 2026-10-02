
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
/// This is top function.
void ctap_r2_0(
  hls::stream< hls::vector< int8_t, 17 > >& v0,
  hls::stream< int16_t >& v1,
  hls::stream< int32_t >& v2
) {	// L2
  int32_t v3 = v2.read();	// L3
  int32_t _st__pid0;	// L4
  _st__pid0 = v3;	// L5
  int32_t v5 = _st__pid0;	// L6
  int32_t slot;	// L7
  slot = v5;	// L8
  bool go;	// L12
  go = 1;	// L13
  while (true) {	// L14
    #pragma HLS pipeline II=1
    #pragma HLS latency max=0
    bool v8 = go;	// L15
    if (!(v8)) break;
    int16_t v9 = v1.read();	// L22
    int16_t y;	// L23
    y = v9;	// L24
    int8_t t[17];	// L28
    for (int v12 = 0; v12 < 17; v12++) {	// L29
      t[v12] = 0;	// L29
    }
    int16_t v13 = y;	// L30
    int32_t v14 = v13;	// L31
    int32_t v15 = v14 & 255;	// L34
    int32_t v16 = v15 ^ 128;	// L37
    ap_int<33> v17 = v16;	// L38
    ap_int<33> v18 = v17 - 128;	// L42
    int8_t v19 = v18;	// L43
    int32_t v20 = slot;	// L44
    int v21 = v20;	// L45
    t[v21] = v19;	// L46
    int16_t v22 = y;	// L47
    int16_t v23 = v22 >> 8;	// L51
    int32_t v24 = v23;	// L52
    int32_t v25 = v24 & 1;	// L55
    int8_t v26 = v25;	// L56
    t[16] = v26;	// L57
    {
      hls::vector< int8_t, 17 > _vec;
      for (int _iv0 = 0; _iv0 < 17; ++_iv0) {
        _vec[_iv0] = t[_iv0];
      }
      v0.write(_vec);
    }	// L58
    int16_t v27 = y;	// L59
    int16_t v28 = v27 >> 8;	// L63
    int32_t v29 = v28;	// L64
    int32_t v30 = v29 & 1;	// L67
    bool v31 = v30 != 0;	// L70
    if (v31) {	// L71
      go = 0;	// L75
    }
  }
}

