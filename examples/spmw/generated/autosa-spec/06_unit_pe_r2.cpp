
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
  hls::stream< int8_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int8_t >& v3,
  hls::stream< int8_t >& v4,
  hls::stream< int32_t >& v5,
  hls::stream< int32_t >& v6
) {	// L2
  int32_t v7 = v0.read();	// L7
  int32_t _st__pid1;	// L8
  _st__pid1 = v7;	// L9
  int32_t row;	// L10
  row = 2;	// L11
  int32_t v10 = _st__pid1;	// L12
  int32_t _col;	// L13
  _col = v10;	// L14
  int32_t acc;	// L15
  acc = 0;	// L16
  l_S_k_0_k: for (int k = 0; k < 4; k++) {	// L17
    int8_t v14 = v1.read();	// L18
    int8_t a;	// L19
    a = v14;	// L20
    int8_t v16 = v2.read();	// L21
    int8_t b;	// L22
    b = v16;	// L23
    int8_t v18 = a;	// L24
    int8_t v19 = b;	// L25
    int16_t v20 = v18;	// L26
    int16_t v21 = v19;	// L27
    int16_t v22 = v20 * v21;	// L28
    int32_t v23 = acc;	// L29
    ap_int<33> v24 = v23;	// L30
    ap_int<33> v25 = v22;	// L31
    ap_int<33> v26 = v24 + v25;	// L32
    int32_t v27 = v26;	// L33
    acc = v27;	// L34
    int8_t v28 = a;	// L35
    v3.write(v28);	// L36
    int8_t v29 = b;	// L37
    v4.write(v29);	// L38
  }
  int32_t v30 = acc;	// L40
  v5.write(v30);	// L41
  int32_t v31 = row;	// L42
  int v32 = v31;	// L43
  for (int v33 = 0; v33 < v32; v33 += 1) {	// L44
    int32_t v34 = v6.read();	// L45
    v5.write(v34);	// L46
  }
}

/// This is top function.
void top(

) {	// L50
  #pragma HLS dataflow
  hls::stream< int32_t > v35;
  #pragma HLS stream variable=v35 depth=2	// L51
  hls::stream< int32_t > v36;
  #pragma HLS stream variable=v36 depth=2	// L52
  hls::stream< int8_t > v37;
  #pragma HLS stream variable=v37 depth=2	// L53
  hls::stream< int8_t > v38;
  #pragma HLS stream variable=v38 depth=2	// L54
  hls::stream< int8_t > v39;
  #pragma HLS stream variable=v39 depth=2	// L55
  hls::stream< int8_t > v40;
  #pragma HLS stream variable=v40 depth=2	// L56
  hls::stream< int32_t > v41;
  #pragma HLS stream variable=v41 depth=1	// L57
  pe_r2_0(v41, v40, v38, v37, v39, v36, v35);	// L58
}

