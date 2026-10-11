
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
/// This is top function.
void mac_r4_0(
  hls::stream< int8_t >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int8_t >& v4,
  hls::stream< int8_t >& v5,
  hls::stream< int32_t >& v6,
  hls::stream< int32_t >& v7
) {	// L2
  int32_t v8 = v6.read();	// L3
  int32_t _st__pid0;	// L4
  _st__pid0 = v8;	// L5
  int32_t v10 = v7.read();	// L6
  int32_t _st__pid1;	// L7
  _st__pid1 = v10;	// L8
  int32_t v12 = _st__pid0;	// L9
  int32_t row;	// L10
  row = v12;	// L11
  int32_t v14 = _st__pid1;	// L12
  int32_t _col;	// L13
  _col = v14;	// L14
  int32_t v16 = row;	// L22
  ap_int<34> v17 = v16;	// L24
  ap_int<34> v18 = 31 - v17;	// L25
  uint8_t v19 = v18;	// L26
  uint8_t below;	// L27
  below = v19;	// L28
  int8_t v21 = v4.read();	// L29
  int8_t w;	// L30
  w = v21;	// L31
  int8_t v23 = below;	// L32
  int v24 = v23;	// L36
  for (int v25 = 0; v25 < v24; v25 += 1) {	// L40
    int8_t v26 = v4.read();	// L41
    v5.write(v26);	// L42
  }
  l_S_m_1_m: for (int m = 0; m < 32; m++) {	// L44
    int8_t v28 = v0.read();	// L45
    int8_t a;	// L46
    a = v28;	// L47
    int32_t v30 = v2.read();	// L48
    int32_t p;	// L49
    p = v30;	// L50
    int32_t v32 = p;	// L51
    int8_t v33 = a;	// L52
    int8_t v34 = w;	// L53
    int16_t v35 = v33;	// L54
    int16_t v36 = v34;	// L55
    int16_t v37 = v35 * v36;	// L56
    ap_int<33> v38 = v32;	// L57
    ap_int<33> v39 = v37;	// L58
    ap_int<33> v40 = v38 + v39;	// L59
    v3.write(v40);	// L60
    int8_t v41 = a;	// L61
    v1.write(v41);	// L62
  }
}

