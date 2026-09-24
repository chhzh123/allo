
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
void lanes4_r0_0(
  hls::stream< hls::vector< int32_t, 2 > >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int32_t >& v4,
  hls::stream< int32_t >& v5,
  hls::stream< int32_t >& v6,
  hls::stream< int32_t >& v7,
  hls::stream< int32_t >& v8
) {	// L2
  int32_t v9[2];
  {
    hls::vector< int32_t, 2 > _vec = v0.read();
    for (int _iv0 = 0; _iv0 < 2; ++_iv0) {
      v9[_iv0] = _vec[_iv0];
    }
  }	// L3
  int32_t v10 = v9[1];	// L4
  int32_t v11 = v10 & 31;	// L6
  int32_t sh;	// L7
  sh = v11;	// L8
  l_S__m_0__m: for (int _m = 0; _m < 256; _m++) {	// L9
  #pragma HLS pipeline II=1
    int32_t v14 = v5.read();	// L10
    int32_t y0;	// L11
    y0 = v14;	// L12
    int32_t v16 = y0;	// L13
    bool v17 = v16 < 0;	// L15
    if (v17) {	// L16
      y0 = 0;	// L17
    }
    int32_t v18 = y0;	// L19
    int32_t v19 = sh;	// L20
    int32_t v20 = v18 >> v19;	// L21
    y0 = v20;	// L22
    int32_t v21 = y0;	// L23
    bool v22 = v21 > 127;	// L25
    if (v22) {	// L26
      y0 = 127;	// L27
    }
    int32_t v23 = y0;	// L29
    v1.write(v23);	// L30
    int32_t v24 = v6.read();	// L31
    int32_t y1;	// L32
    y1 = v24;	// L33
    int32_t v26 = y1;	// L34
    bool v27 = v26 < 0;	// L35
    if (v27) {	// L36
      y1 = 0;	// L37
    }
    int32_t v28 = y1;	// L39
    int32_t v29 = sh;	// L40
    int32_t v30 = v28 >> v29;	// L41
    y1 = v30;	// L42
    int32_t v31 = y1;	// L43
    bool v32 = v31 > 127;	// L44
    if (v32) {	// L45
      y1 = 127;	// L46
    }
    int32_t v33 = y1;	// L48
    v2.write(v33);	// L49
    int32_t v34 = v7.read();	// L50
    int32_t y2;	// L51
    y2 = v34;	// L52
    int32_t v36 = y2;	// L53
    bool v37 = v36 < 0;	// L54
    if (v37) {	// L55
      y2 = 0;	// L56
    }
    int32_t v38 = y2;	// L58
    int32_t v39 = sh;	// L59
    int32_t v40 = v38 >> v39;	// L60
    y2 = v40;	// L61
    int32_t v41 = y2;	// L62
    bool v42 = v41 > 127;	// L63
    if (v42) {	// L64
      y2 = 127;	// L65
    }
    int32_t v43 = y2;	// L67
    v3.write(v43);	// L68
    int32_t v44 = v8.read();	// L69
    int32_t y3;	// L70
    y3 = v44;	// L71
    int32_t v46 = y3;	// L72
    bool v47 = v46 < 0;	// L73
    if (v47) {	// L74
      y3 = 0;	// L75
    }
    int32_t v48 = y3;	// L77
    int32_t v49 = sh;	// L78
    int32_t v50 = v48 >> v49;	// L79
    y3 = v50;	// L80
    int32_t v51 = y3;	// L81
    bool v52 = v51 > 127;	// L82
    if (v52) {	// L83
      y3 = 127;	// L84
    }
    int32_t v53 = y3;	// L86
    v4.write(v53);	// L87
  }
}

