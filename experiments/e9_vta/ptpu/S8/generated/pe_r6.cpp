
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
void pe_r6_0(
  hls::stream< int16_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int8_t >& v2
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
    bool v7 = go;	// L24
    if (!(v7)) break;
    int16_t v8 = v0.read();	// L31
    int16_t t;	// L32
    t = v8;	// L33
    bool v10 = live;	// L34
    if (v10) {	// L39
      int16_t v11 = t;	// L40
      int32_t v12 = v11;	// L41
      int32_t v13 = v12 & 255;	// L44
      int32_t v14 = v13 ^ 128;	// L47
      ap_int<33> v15 = v14;	// L48
      ap_int<33> v16 = v15 - 128;	// L52
      int8_t v17 = v16;	// L53
      int8_t a;	// L54
      a = v17;	// L55
      int32_t p;	// L58
      p = 0;	// L59
      int32_t v20 = p;	// L60
      int8_t v21 = a;	// L61
      int8_t v22 = cur;	// L62
      int16_t v23 = v21;	// L63
      int16_t v24 = v22;	// L64
      int16_t v25 = v23 * v24;	// L65
      #pragma HLS bind_op variable=v25 op=mul impl=fabric
      ap_int<33> v26 = v20;	// L66
      ap_int<33> v27 = v25;	// L67
      ap_int<33> v28 = v26 + v27;	// L68
      v1.write(v28);	// L69
    }
    int8_t v29 = v2.read();	// L71
    nxt = v29;	// L72
    int16_t v30 = t;	// L73
    int16_t v31 = v30 >> 8;	// L77
    int32_t v32 = v31;	// L78
    int32_t v33 = v32 & 1;	// L81
    bool v34 = v33 != 0;	// L84
    if (v34) {	// L85
      int8_t v35 = nxt;	// L86
      cur = v35;	// L87
      live = 1;	// L91
    }
    int16_t v36 = t;	// L93
    int16_t v37 = v36 >> 9;	// L97
    int32_t v38 = v37;	// L98
    int32_t v39 = v38 & 1;	// L101
    bool v40 = v39 != 0;	// L104
    if (v40) {	// L105
      go = 0;	// L109
    }
  }
}

