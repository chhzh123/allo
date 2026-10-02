
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
void pe_r0_0(
  hls::stream< int16_t >& v0,
  hls::stream< int16_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int32_t >& v3,
  hls::stream< int8_t >& v4,
  hls::stream< int8_t >& v5
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
    bool v10 = go;	// L24
    if (!(v10)) break;
    int16_t v11 = v0.read();	// L31
    int16_t t;	// L32
    t = v11;	// L33
    int16_t v13 = t;	// L34
    v1.write(v13);	// L35
    bool v14 = live;	// L36
    if (v14) {	// L41
      int16_t v15 = t;	// L42
      int32_t v16 = v15;	// L43
      int32_t v17 = v16 & 255;	// L46
      int32_t v18 = v17 ^ 128;	// L49
      ap_int<33> v19 = v18;	// L50
      ap_int<33> v20 = v19 - 128;	// L54
      int8_t v21 = v20;	// L55
      int8_t a;	// L56
      a = v21;	// L57
      int32_t v23 = v2.read();	// L58
      int32_t p;	// L59
      p = v23;	// L60
      int32_t v25 = p;	// L61
      int8_t v26 = a;	// L62
      int8_t v27 = cur;	// L63
      int16_t v28 = v26;	// L64
      int16_t v29 = v27;	// L65
      int16_t v30 = v28 * v29;	// L66
      #pragma HLS bind_op variable=v30 op=mul impl=fabric
      ap_int<33> v31 = v25;	// L67
      ap_int<33> v32 = v30;	// L68
      ap_int<33> v33 = v31 + v32;	// L69
      v3.write(v33);	// L70
    }
    int8_t v34 = nxt;	// L72
    v5.write(v34);	// L73
    int8_t v35 = v4.read();	// L74
    nxt = v35;	// L75
    int16_t v36 = t;	// L76
    int16_t v37 = v36 >> 8;	// L80
    int32_t v38 = v37;	// L81
    int32_t v39 = v38 & 1;	// L84
    bool v40 = v39 != 0;	// L87
    if (v40) {	// L88
      int8_t v41 = nxt;	// L89
      cur = v41;	// L90
      live = 1;	// L94
    }
    int16_t v42 = t;	// L96
    int16_t v43 = v42 >> 9;	// L100
    int32_t v44 = v43;	// L101
    int32_t v45 = v44 & 1;	// L104
    bool v46 = v45 != 0;	// L107
    if (v46) {	// L108
      go = 0;	// L112
    }
  }
}

