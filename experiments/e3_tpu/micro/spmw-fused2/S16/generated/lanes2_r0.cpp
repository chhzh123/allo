
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
void lanes2_r0_0(
  hls::stream< hls::vector< int32_t, 2 > >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int32_t >& v4
) {	// L2
  int32_t v5[2];
  {
    hls::vector< int32_t, 2 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v5[_iv0] = _vec[_iv0];
    }
  }	// L3
  int32_t v6 = v5[1];	// L4
  int32_t v7 = v6 & 31;	// L6
  int32_t sh;	// L7
  sh = v7;	// L8
  l_S__m_0__m: for (int _m = 0; _m < 256; _m++) {	// L9
  #pragma HLS pipeline II=1
    int32_t v10 = v3.read();	// L10
    int32_t y0;	// L11
    y0 = v10;	// L12
    int32_t v12 = y0;	// L13
    bool v13 = v12 < 0;	// L15
    if (v13) {	// L16
      y0 = 0;	// L17
    }
    int32_t v14 = y0;	// L19
    int32_t v15 = sh;	// L20
    int32_t v16 = v14 >> v15;	// L21
    y0 = v16;	// L22
    int32_t v17 = y0;	// L23
    bool v18 = v17 > 127;	// L25
    if (v18) {	// L26
      y0 = 127;	// L27
    }
    int32_t v19 = y0;	// L29
    v1.write(v19);	// L30
    int32_t v20 = v4.read();	// L31
    int32_t y1;	// L32
    y1 = v20;	// L33
    int32_t v22 = y1;	// L34
    bool v23 = v22 < 0;	// L35
    if (v23) {	// L36
      y1 = 0;	// L37
    }
    int32_t v24 = y1;	// L39
    int32_t v25 = sh;	// L40
    int32_t v26 = v24 >> v25;	// L41
    y1 = v26;	// L42
    int32_t v27 = y1;	// L43
    bool v28 = v27 > 127;	// L44
    if (v28) {	// L45
      y1 = 127;	// L46
    }
    int32_t v29 = y1;	// L48
    v2.write(v29);	// L49
  }
}

