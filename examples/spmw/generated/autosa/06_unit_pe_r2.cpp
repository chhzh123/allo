
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
void pe_r2_0(
  hls::stream< int32_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int8_t >& v3,
  hls::stream< int8_t >& v4,
  hls::stream< int32_t >& v5,
  hls::stream< int32_t >& v6
) {	// L2
  int32_t v7 = v5.read();	// L3
  int32_t _st__pid0;	// L4
  _st__pid0 = v7;	// L5
  int32_t v9 = v6.read();	// L6
  int32_t _st__pid1;	// L7
  _st__pid1 = v9;	// L8
  int32_t v11 = _st__pid0;	// L9
  int32_t row;	// L10
  row = v11;	// L11
  int32_t v13 = _st__pid1;	// L12
  int32_t _col;	// L13
  _col = v13;	// L14
  int32_t acc;	// L17
  acc = 0;	// L18
  l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L19
    int8_t v17 = v4.read();	// L20
    int8_t a;	// L21
    a = v17;	// L22
    int8_t v19 = v2.read();	// L23
    int8_t b;	// L24
    b = v19;	// L25
    int8_t v21 = a;	// L26
    int8_t v22 = b;	// L27
    int16_t v23 = v21;	// L28
    int16_t v24 = v22;	// L29
    int16_t v25 = v23 * v24;	// L30
    int32_t v26 = acc;	// L31
    ap_int<33> v27 = v26;	// L32
    ap_int<33> v28 = v25;	// L33
    ap_int<33> v29 = v27 + v28;	// L34
    int32_t v30 = v29;	// L35
    acc = v30;	// L36
    int8_t v31 = b;	// L37
    v3.write(v31);	// L38
  }
  int32_t v32 = acc;	// L40
  v1.write(v32);	// L41
  int32_t v33 = row;	// L42
  int v34 = v33;	// L46
  for (int v35 = 0; v35 < v34; v35 += 1) {	// L50
    int32_t v36 = v0.read();	// L51
    v1.write(v36);	// L52
  }
}

