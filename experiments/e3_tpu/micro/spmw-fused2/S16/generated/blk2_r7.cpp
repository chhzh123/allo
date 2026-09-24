
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
void blk2_r7_0(
  hls::stream< int8_t >& v0,
  hls::stream< int8_t >& v1,
  hls::stream< int8_t >& v2,
  hls::stream< int8_t >& v3,
  hls::stream< int32_t >& v4,
  hls::stream< int32_t >& v5,
  hls::stream< int32_t >& v6,
  hls::stream< int32_t >& v7,
  hls::stream< int8_t >& v8,
  hls::stream< int8_t >& v9,
  hls::stream< int8_t >& v10,
  hls::stream< int8_t >& v11
) {	// L2
  int8_t n0_0;	// L5
  n0_0 = 0;	// L6
  int8_t n0_1;	// L7
  n0_1 = 0;	// L8
  int8_t n1_0;	// L9
  n1_0 = 0;	// L10
  int8_t n1_1;	// L11
  n1_1 = 0;	// L12
  l_S__k_0__k: for (int _k = 0; _k < 16; _k++) {	// L13
  #pragma HLS pipeline II=1
    int8_t v17 = n0_1;	// L14
    v9.write(v17);	// L15
    int8_t v18 = n0_0;	// L16
    n0_1 = v18;	// L17
    int8_t v19 = v8.read();	// L18
    n0_0 = v19;	// L19
    int8_t v20 = n1_1;	// L20
    v11.write(v20);	// L21
    int8_t v21 = n1_0;	// L22
    n1_1 = v21;	// L23
    int8_t v22 = v10.read();	// L24
    n1_0 = v22;	// L25
  }
  int8_t v23 = n0_0;	// L27
  int8_t c0_0;	// L28
  c0_0 = v23;	// L29
  int8_t v25 = n0_1;	// L30
  int8_t c0_1;	// L31
  c0_1 = v25;	// L32
  int8_t v27 = n1_0;	// L33
  int8_t c1_0;	// L34
  c1_0 = v27;	// L35
  int8_t v29 = n1_1;	// L36
  int8_t c1_1;	// L37
  c1_1 = v29;	// L38
  l_S_s_1_s: for (int s = 0; s < 256; s++) {	// L39
  #pragma HLS pipeline II=1
    int8_t v32 = v0.read();	// L40
    int8_t a0;	// L41
    a0 = v32;	// L42
    int8_t v34 = v2.read();	// L43
    int8_t a1;	// L44
    a1 = v34;	// L45
    int32_t v36 = v4.read();	// L46
    int32_t p0;	// L47
    p0 = v36;	// L48
    int32_t v38 = v6.read();	// L49
    int32_t p1;	// L50
    p1 = v38;	// L51
    int8_t v40 = a0;	// L52
    v1.write(v40);	// L53
    int8_t v41 = a1;	// L54
    v3.write(v41);	// L55
    int32_t v42 = p0;	// L56
    int8_t v43 = a0;	// L57
    int8_t v44 = c0_0;	// L58
    int16_t v45 = v43;	// L59
    int16_t v46 = v44;	// L60
    int16_t v47 = v45 * v46;	// L61
    #pragma HLS bind_op variable=v47 op=mul impl=fabric
    int8_t v48 = a1;	// L62
    int8_t v49 = c1_0;	// L63
    int16_t v50 = v48;	// L64
    int16_t v51 = v49;	// L65
    int16_t v52 = v50 * v51;	// L66
    #pragma HLS bind_op variable=v52 op=mul impl=fabric
    ap_int<17> v53 = v47;	// L67
    ap_int<17> v54 = v52;	// L68
    ap_int<17> v55 = v53 + v54;	// L69
    ap_int<33> v56 = v42;	// L70
    ap_int<33> v57 = v55;	// L71
    ap_int<33> v58 = v56 + v57;	// L72
    v5.write(v58);	// L73
    int32_t v59 = p1;	// L74
    int8_t v60 = a0;	// L75
    int8_t v61 = c0_1;	// L76
    int16_t v62 = v60;	// L77
    int16_t v63 = v61;	// L78
    int16_t v64 = v62 * v63;	// L79
    #pragma HLS bind_op variable=v64 op=mul impl=fabric
    int8_t v65 = a1;	// L80
    int8_t v66 = c1_1;	// L81
    int16_t v67 = v65;	// L82
    int16_t v68 = v66;	// L83
    int16_t v69 = v67 * v68;	// L84
    #pragma HLS bind_op variable=v69 op=mul impl=fabric
    ap_int<17> v70 = v64;	// L85
    ap_int<17> v71 = v69;	// L86
    ap_int<17> v72 = v70 + v71;	// L87
    ap_int<33> v73 = v59;	// L88
    ap_int<33> v74 = v72;	// L89
    ap_int<33> v75 = v73 + v74;	// L90
    v7.write(v75);	// L91
    int8_t v76 = n0_1;	// L92
    v9.write(v76);	// L93
    int8_t v77 = n0_0;	// L94
    n0_1 = v77;	// L95
    int8_t v78 = v8.read();	// L96
    n0_0 = v78;	// L97
    int8_t v79 = n1_1;	// L98
    v11.write(v79);	// L99
    int8_t v80 = n1_0;	// L100
    n1_1 = v80;	// L101
    int8_t v81 = v10.read();	// L102
    n1_0 = v81;	// L103
    int32_t v82 = s;	// L104
    int32_t v83 = v82 & 15;	// L106
    bool v84 = v83 == 15;	// L107
    if (v84) {	// L108
      int8_t v85 = n0_0;	// L109
      c0_0 = v85;	// L110
      int8_t v86 = n0_1;	// L111
      c0_1 = v86;	// L112
      int8_t v87 = n1_0;	// L113
      c1_0 = v87;	// L114
      int8_t v88 = n1_1;	// L115
      c1_1 = v88;	// L116
    }
  }
}

