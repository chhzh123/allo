
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
using namespace std;
void pe_r2_0(
  hls::stream< int32_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int8_t >& v3,
  hls::stream< int8_t >& v4,
  hls::stream< int32_t >& v5,
  hls::stream< int32_t >& v6
) {	// L2
  int32_t v7 = v0.read();	// L6
  int32_t _st__pid0;	// L7
  _st__pid0 = v7;	// L8
  int32_t v9 = v1.read();	// L9
  int32_t _st__pid1;	// L10
  _st__pid1 = v9;	// L11
  int32_t v11 = _st__pid0;	// L12
  int32_t row;	// L13
  row = v11;	// L14
  int32_t v13 = _st__pid1;	// L15
  int32_t _col;	// L16
  _col = v13;	// L17
  int32_t acc;	// L18
  acc = 0;	// L19
  l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L20
    int8_t v17 = v2.read();	// L21
    int8_t a;	// L22
    a = v17;	// L23
    int8_t v19 = v3.read();	// L24
    int8_t b;	// L25
    b = v19;	// L26
    int8_t v21 = a;	// L27
    int8_t v22 = b;	// L28
    int16_t v23 = v21;	// L29
    int16_t v24 = v22;	// L30
    int16_t v25 = v23 * v24;	// L31
    int32_t v26 = acc;	// L32
    ap_int<33> v27 = v26;	// L33
    ap_int<33> v28 = v25;	// L34
    ap_int<33> v29 = v27 + v28;	// L35
    int32_t v30 = v29;	// L36
    acc = v30;	// L37
    int8_t v31 = b;	// L38
    v4.write(v31);	// L39
  }
  int32_t v32 = acc;	// L41
  v5.write(v32);	// L42
  int32_t v33 = row;	// L43
  int v34 = v33;	// L44
  for (int v35 = 0; v35 < v34; v35 += 1) {	// L45
    int32_t v36 = v6.read();	// L46
    v5.write(v36);	// L47
  }
}

/// This is top function.
void top(

) {	// L51
  #pragma HLS dataflow
  hls::stream< int32_t > v37;
  #pragma HLS stream variable=v37 depth=2	// L52
  hls::stream< int32_t > v38;
  #pragma HLS stream variable=v38 depth=2	// L53
  hls::stream< int8_t > v39;
  #pragma HLS stream variable=v39 depth=2	// L54
  hls::stream< int8_t > v40;
  #pragma HLS stream variable=v40 depth=2	// L55
  hls::stream< int8_t > v41;
  #pragma HLS stream variable=v41 depth=2	// L56
  hls::stream< int32_t > v42;
  #pragma HLS stream variable=v42 depth=1	// L57
  hls::stream< int32_t > v43;
  #pragma HLS stream variable=v43 depth=1	// L58
  pe_r2_0(v42, v43, v41, v39, v40, v38, v37);	// L59
}

