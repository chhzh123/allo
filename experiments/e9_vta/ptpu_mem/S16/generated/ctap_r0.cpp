
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
void ctap_r0_0(
  hls::stream< hls::vector< int8_t, 17 > >& v0,
  hls::stream< hls::vector< int8_t, 17 > >& v1,
  hls::stream< int16_t >& v2,
  hls::stream< int32_t >& v3
) {	// L2
  int32_t v4 = v3.read();	// L3
  int32_t _st__pid0;	// L4
  _st__pid0 = v4;	// L5
  int32_t v6 = _st__pid0;	// L6
  int32_t slot;	// L7
  slot = v6;	// L8
  bool go;	// L12
  go = 1;	// L13
  while (true) {	// L14
    #pragma HLS pipeline II=1
    #pragma HLS latency max=0
    bool v9 = go;	// L15
    if (!(v9)) break;
    int16_t v10 = v2.read();	// L22
    int16_t y;	// L23
    y = v10;	// L24
    int8_t v12[17];
    {
      hls::vector< int8_t, 17 > _vec = v0.read();
      for (int _iv0 = 0; _iv0 < 17; ++_iv0) {
        v12[_iv0] = _vec[_iv0];
      }
    }	// L25
    int16_t v13 = y;	// L26
    int32_t v14 = v13;	// L27
    int32_t v15 = v14 & 255;	// L30
    int32_t v16 = v15 ^ 128;	// L33
    ap_int<33> v17 = v16;	// L34
    ap_int<33> v18 = v17 - 128;	// L38
    int8_t v19 = v18;	// L39
    int32_t v20 = slot;	// L40
    int v21 = v20;	// L41
    v12[v21] = v19;	// L42
    int16_t v22 = y;	// L43
    int16_t v23 = v22 >> 8;	// L47
    int32_t v24 = v23;	// L48
    int32_t v25 = v24 & 1;	// L51
    int8_t v26 = v25;	// L52
    v12[16] = v26;	// L53
    {
      hls::vector< int8_t, 17 > _vec;
      for (int _iv0 = 0; _iv0 < 17; ++_iv0) {
        _vec[_iv0] = v12[_iv0];
      }
      v1.write(_vec);
    }	// L54
    int16_t v27 = y;	// L55
    int16_t v28 = v27 >> 8;	// L59
    int32_t v29 = v28;	// L60
    int32_t v30 = v29 & 1;	// L63
    bool v31 = v30 != 0;	// L66
    if (v31) {	// L67
      go = 0;	// L71
    }
  }
}

