
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
void pe_r3_0(
  hls::stream< int16_t >& v0,
  hls::stream< int32_t >& v1,
  hls::stream< int32_t >& v2,
  hls::stream< int8_t >& v3
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
    bool v8 = go;	// L24
    if (!(v8)) break;
    int16_t v9 = v0.read();	// L31
    int16_t t;	// L32
    t = v9;	// L33
    bool v11 = live;	// L34
    if (v11) {	// L39
      int16_t v12 = t;	// L40
      int32_t v13 = v12;	// L41
      int32_t v14 = v13 & 255;	// L44
      int32_t v15 = v14 ^ 128;	// L47
      ap_int<33> v16 = v15;	// L48
      ap_int<33> v17 = v16 - 128;	// L52
      int8_t v18 = v17;	// L53
      int8_t a;	// L54
      a = v18;	// L55
      int32_t v20 = v1.read();	// L56
      int32_t p;	// L57
      p = v20;	// L58
      int32_t v22 = p;	// L59
      int8_t v23 = a;	// L60
      int8_t v24 = cur;	// L61
      int16_t v25 = v23;	// L62
      int16_t v26 = v24;	// L63
      int16_t v27 = v25 * v26;	// L64
      #pragma HLS bind_op variable=v27 op=mul impl=fabric
      ap_int<33> v28 = v22;	// L65
      ap_int<33> v29 = v27;	// L66
      ap_int<33> v30 = v28 + v29;	// L67
      v2.write(v30);	// L68
    }
    int8_t v31 = v3.read();	// L70
    nxt = v31;	// L71
    int16_t v32 = t;	// L72
    int16_t v33 = v32 >> 8;	// L76
    int32_t v34 = v33;	// L77
    int32_t v35 = v34 & 1;	// L80
    bool v36 = v35 != 0;	// L83
    if (v36) {	// L84
      int8_t v37 = nxt;	// L85
      cur = v37;	// L86
      live = 1;	// L90
    }
    int16_t v38 = t;	// L92
    int16_t v39 = v38 >> 9;	// L96
    int32_t v40 = v39;	// L97
    int32_t v41 = v40 & 1;	// L100
    bool v42 = v41 != 0;	// L103
    if (v42) {	// L104
      go = 0;	// L108
    }
  }
}

