
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
void pe_r8_0(
  hls::stream< int16_t >& v0,
  hls::stream< int16_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int8_t >& v3,
  hls::stream< int8_t >& v4
) {	// L2
  int8_t nxt;	// L6
  nxt = 0;	// L7
  int8_t cur;	// L11
  cur = 0;	// L12
  bool live;	// L16
  live = 0;	// L17
  bool go;	// L21
  go = 1;	// L22
  while (true) {	// L23
    #pragma HLS pipeline II=1
    #pragma HLS latency max=0
    bool v9 = go;	// L24
    if (!(v9)) break;
    int16_t v10 = v0.read();	// L31
    int16_t t;	// L32
    t = v10;	// L33
    int16_t v12 = t;	// L34
    v1.write(v12);	// L35
    bool v13 = live;	// L36
    if (v13) {	// L41
      int16_t v14 = t;	// L42
      int32_t v15 = v14;	// L43
      int32_t v16 = v15 & 255;	// L46
      int32_t v17 = v16 ^ 128;	// L49
      ap_int<33> v18 = v17;	// L50
      ap_int<33> v19 = v18 - 128;	// L54
      int8_t v20 = v19;	// L55
      int8_t a;	// L56
      a = v20;	// L57
      int32_t p;	// L60
      p = 0;	// L61
      int32_t v23 = p;	// L62
      int8_t v24 = a;	// L63
      int8_t v25 = cur;	// L64
      int16_t v26 = v24;	// L65
      int16_t v27 = v25;	// L66
      int16_t v28 = v26 * v27;	// L67
      #pragma HLS bind_op variable=v28 op=mul impl=fabric
      ap_int<33> v29 = v23;	// L68
      ap_int<33> v30 = v28;	// L69
      ap_int<33> v31 = v29 + v30;	// L70
      v2.write(v31);	// L71
    }
    int8_t v32 = nxt;	// L73
    v4.write(v32);	// L74
    int8_t v33 = v3.read();	// L75
    nxt = v33;	// L76
    int16_t v34 = t;	// L77
    int16_t v35 = v34 >> 8;	// L81
    int32_t v36 = v35;	// L82
    int32_t v37 = v36 & 1;	// L85
    bool v38 = v37 != 0;	// L88
    if (v38) {	// L89
      int8_t v39 = nxt;	// L90
      cur = v39;	// L91
      live = 1;	// L95
    }
    int16_t v40 = t;	// L97
    int16_t v41 = v40 >> 9;	// L101
    int32_t v42 = v41;	// L102
    int32_t v43 = v42 & 1;	// L105
    bool v44 = v43 != 0;	// L108
    if (v44) {	// L109
      go = 0;	// L113
    }
  }
}

